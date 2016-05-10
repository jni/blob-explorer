"""
Compute features on a bunch of images and display as hoverable-scatterplot.
"""
import os
import sys
import tempfile
import pathlib
from io import BytesIO
import base64

import numpy as np
from PIL import Image

from matplotlib import cm

from scipy import ndimage as ndi
from skimage import io, filters, measure, morphology, img_as_ubyte
import pandas as pd
from sklearn import decomposition, manifold

from bokeh.models import (LassoSelectTool, PanTool,
                          ResizeTool, ResetTool,
                          HoverTool, WheelZoomTool)
TOOLS = [LassoSelectTool, PanTool, WheelZoomTool, ResizeTool, ResetTool]
from bokeh.models import ColumnDataSource
from bokeh import plotting as bplot


def to_png(arr):
    out = BytesIO()
    im = Image.fromarray(arr)
    im.save(out, format='png')
    return out.getvalue()


def extract_properties(image, closing_size=2):
    selem = morphology.disk(radius=closing_size)
    thresholded = image > filters.threshold_otsu(image)
    closed = morphology.binary_closing(thresholded, selem)
    regions = ndi.label(closed)[0]
    propnames = ['area', 'convex_area', 'eccentricity', 'euler_number',
                 'extent', 'min_intensity', 'mean_intensity', 'max_intensity',
                 'minor_axis_length', 'major_axis_length']
    props = measure.regionprops(regions, image)
    data_table = []
    for obj in props:
        data_point = [getattr(obj, p) for p in propnames]
        data_table.append(data_point)
    return propnames, np.array(data_table), props


def extract_properties_multi_image(image_collection, closing_size=2,
                                   min_blob_size=9, max_blob_size=100):
    all_results = []
    all_objs = []
    times = []
    filenames = []
    for idx, image in enumerate(image_collection):
        print('processing image ', idx)
        filename = image_collection.files[idx]
        timepoint = int(filename.split('-')[1][:-1])
        names, proptable, objs = extract_properties(image, closing_size)
        passed = np.flatnonzero((proptable[:, 0] > min_blob_size) *
                                (proptable[:, 0] < max_blob_size))
        all_results.append(proptable[passed])
        all_objs.extend([objs[i] for i in passed])
        times.extend([timepoint] * len(passed))
        filenames.extend([filename] * len(passed))
    all_results = np.vstack(all_results)
    times = np.array(times)[:, np.newaxis]
    dec, dec_names, pca_weights = dimension_reductions(all_results)
    col_names = ['time'] + names + dec_names
    data = np.hstack((times, all_results, dec))
    df = pd.DataFrame(data, columns=col_names)
    df['images'] = [obj.intensity_image for obj in all_objs]
    df['source_filenames'] = [os.path.split(p)[1] for p in filenames]
    return all_results, df, pca_weights


def dimension_reductions(data_table):
    """Perform various 2D projections of the data.

    Parameters
    ----------
    data_table : array of float, shape (n_samples, n_features)
        The input data.

    Returns
    -------
    vecs : array of float, shape (n_samples, 6)
        Three 2D projections of the data:
            - PCA
            - tSNE
    names : list of string
        The names of the returned columns.
    components : array of float, shape (2, n_features)
        The PCA vector loadings.
    """
    mean = np.mean(data_table, axis=0)
    std = np.std(data_table, axis=0)
    norm_data = (data_table - mean) / std
    pca_obj = decomposition.PCA(n_components=2)
    pca = pca_obj.fit_transform(norm_data)
    tsne = manifold.TSNE().fit_transform(norm_data)
    names = ['PC1', 'PC2', 'tSNE-0', 'tSNE-1']
    return np.hstack((pca, tsne)), names, pca_obj.components_


def b64_image_files(images, colormap='magma'):
    cmap = cm.get_cmap(colormap)
    urls = []
    for im in images:
        png = to_png(img_as_ubyte(cmap(im)))
        url = 'data:image/png;base64,' + base64.b64encode(png).decode('utf-8')
    return urls


def bokeh_plot(df):
    tooltip = """
        <div>
            <div>
                <img
                src="@image_files" height="60" alt="image"
                style="float: left; margin: 0px 15px 15px 0px; image-rendering: pixelated;"
                border="2"
                ></img>
            </div>
            <div>
                <span style="font-size: 17px;">@source_filenames</span>
            </div>
        </div>
              """
    filenames = b64_image_files(df['images'])
    df['image_files'] = filenames
    colors_raw = cm.viridis((df['time'] - df['time'].min()) /
                            (df['time'].max() - df['time'].min()), bytes=True)
    colors_str = ['#%02x%02x%02x' % tuple(c[:3]) for c in colors_raw]
    df['color'] = colors_str
    source = ColumnDataSource(df)
    bplot.output_file('plot.html')
    hover0 = HoverTool(tooltips=tooltip)
    hover1 = HoverTool(tooltips=tooltip)
    tools0 = [t() for t in TOOLS] + [hover0]
    tools1 = [t() for t in TOOLS] + [hover1]
    pca = bplot.figure(tools=tools0)
    pca.circle('PC1', 'PC2', color='color', source=source)
    tsne = bplot.figure(tools=tools1)
    tsne.circle('tSNE-0', 'tSNE-1', color='color', source=source)
    p = bplot.gridplot([[pca, tsne]])
    bplot.show(p)


def normalize_images(ims):
    max_val = np.median([np.percentile(im, 99.9) for im in ims])
    for im in ims:
        im /= max_val
        np.clip(im, 0, 1, out=im)
    return ims


def main(argv):
    print('reading images')
    images = io.imread_collection(argv[1:],
                                  conserve_memory=False, plugin='tifffile')
    images = normalize_images(images)
    print('extracting data')
    table, df, weights = extract_properties_multi_image(images)

    print('preparing plots')
    bokeh_plot(df)


if __name__ == '__main__':
    main(sys.argv)
