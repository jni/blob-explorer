"""
compute the mean and stddev of 100 data sets and plot mean vs stddev.
When you click on one of the mu, sigma points, plot the raw data from
the dataset that generated the mean and stddev
"""
import sys
import numpy as np

import matplotlib as mpl
mpl.use('qt4agg')
import matplotlib.pyplot as plt
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'magma'
plt.style.use('seaborn-colorblind')


from scipy import ndimage as ndi
from skimage import io, filters, measure, morphology
import pandas as pd
from sklearn import decomposition, manifold


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
                                   min_blob_size=4, max_blob_size=100):
    all_results = []
    all_objs = []
    times = []
    for idx, image in enumerate(image_collection):
        print('processing image ', idx)
        timepoint = int(image_collection.files[idx].split('-')[1][:-1])
        names, proptable, objs = extract_properties(image, closing_size)
        passed = np.flatnonzero((proptable[:, 0] > min_blob_size) *
                                (proptable[:, 0] < max_blob_size))
        all_results.append(proptable[passed])
        all_objs.extend([objs[i] for i in passed])
        times.extend([timepoint] * len(passed))
    all_results = np.vstack(all_results)
    times = np.array(times)[:, np.newaxis]
    dec, dec_names, pca_weights = dimension_reductions(all_results)
    col_names = ['time'] + names + dec_names
    data = np.hstack((times, all_results, dec))
    df = pd.DataFrame(data, columns=col_names)
    df['images'] = [obj.intensity_image for obj in all_objs]
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


if __name__ == '__main__':
    print('reading images')
    images = io.imread_collection(sys.argv[1:],
                                  conserve_memory=False, plugin='tifffile')
    print('extracting data')
    table, df, weights = extract_properties_multi_image(images)

    print('preparing plots')
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
    pts_ax, im_ax = axes.ravel()
    pts_ax.set_title('Images grouped by similarity')
    im_ax.set_title('Clicked image')
    x, y = 'tSNE-0', 'tSNE-1'
    if len(sys.argv) > 1 and sys.argv[1].lower() == 'pca':
        x, y = 'PC1', 'PC2'
    pts_ax.scatter(df[x], df[y], c=df['time'], picker=5)

    def onpick(event):

        N = len(event.ind)
        if N == 0:
            return True
        dataind = event.ind[0]  # pick only the first element, ignore others
        image = df['images'][dataind]
        im_ax.imshow(image)
        im_ax.set_yticks([0, image.shape[0] - 1])
        im_ax.set_xticks([0, image.shape[1] - 1])
        return True

    fig.canvas.mpl_connect('pick_event', onpick)

    plt.show()
