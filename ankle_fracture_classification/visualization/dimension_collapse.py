import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import linalg
from sklearn.covariance import EmpiricalCovariance
from sklearn.decomposition import TruncatedSVD


METHOD = "numpy"  # "numpy" or "scipy

# Load PyTorch file (assuming it's a tensor saved as .pt or .pth)
DIRS = [
    # "visualization/projected_feature_vectors/moco_v2/moco_v2_coco",
    # "visualization/projected_feature_vectors/moco_v3/moco_v3_coco",
    # "visualization/projected_feature_vectors/vicreg/vicreg_coco",
    # "visualization/projected_feature_vectors/vicregl/vicregl_coco",
    # "visualization/projected_feature_vectors/moco_v2/moco_v2_downstream_v2",
    # "visualization/projected_feature_vectors/moco_v3/moco_v3_downstream_v2",
    # "visualization/projected_feature_vectors/vicreg/vicreg_downstream_v2",
    # "visualization/projected_feature_vectors/vicregl/vicregl_downstream_v2",
    "visualization/projected_feature_vectors/dino/dinov1_luxry",
    # "visualization/projected_feature_vectors/dino/dinov1_imagenet_train",
    # "visualization/projected_feature_vectors/dino/dinov1_downstream_v2",
    # "visualization/projected_feature_vectors/dino/dinov1_coco",
]


for dir in tqdm(DIRS):
    vectors = torch.load(dir + "/feature_vectors.pt")

    # Convert to NumPy for subsequent operations
    vectors_np = vectors.numpy()
    vectors_np = vectors_np.T
    vectors_np = vectors_np[:10000]

    if METHOD == "numpy":
        # Covariance matrix calculation
        cov_matrix = np.cov(vectors_np)
        # Singular Value Decomposition
        U, S, Vh = np.linalg.svd(cov_matrix)
    elif METHOD == "scipy":
        cov_matrix = np.cov(vectors_np)
        U, S, Vh = linalg.svd(cov_matrix)
    elif METHOD == "sklearn":
        vectors_np = vectors_np.T
        cov = EmpiricalCovariance().fit(vectors_np)
        cov_matrix = cov.covariance_
        svd = TruncatedSVD(n_components=cov_matrix.shape[1] - 1)
        svd.fit(vectors_np)
        U = svd.components_
        S = svd.singular_values_
        Vh = svd.components_

    # Logarithm of singular values
    log_S = np.log(S)
    print("Number of singular values:", len(S))
    np.save(dir + "/singular_values.npy", S)

    # Plotting
    plt.plot(np.arange(len(log_S)), log_S)
    plt.xlabel("Index of Singular Value")
    plt.ylabel("Logarithm of Singular Value")
    plt.title("Singular Value Spectrum (Logarithmic)")
    plt.grid(axis="y")
    # plt.plot()
    plt.savefig(dir + "/singular_value_spectrum_test.png")
    plt.close()

    del vectors, vectors_np, cov_matrix, U, S, Vh, log_S
