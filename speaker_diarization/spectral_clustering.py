from spectralcluster import SpectralClusterer
from spectralcluster import RefinementOptions
from spectralcluster import DEFAULT_REFINEMENT_SEQUENCE

refinement_options = RefinementOptions(
    gaussian_blur_sigma=1,
    p_percentile=0.95,
    thresholding_soft_multiplier=0.01,
    thresholding_with_row_max=True,
    refinement_sequence=DEFAULT_REFINEMENT_SEQUENCE)

clusterer = SpectralClusterer(
    min_clusters=2,
    max_clusters=7,
    autotune=None,
    laplacian_type=None,
    refinement_options=refinement_options,
    custom_dist="cosine")

def find_clusters(x):
    return clusterer.predict(X)