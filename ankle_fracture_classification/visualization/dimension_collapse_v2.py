import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

METHOD = "vicreg"
DIR = "visualization/projected_feature_vectors/" + METHOD
LOG = True

# Load PyTorch file (assuming it's a tensor saved as .pt or .pth)
train_vectors = torch.load(DIR + "/" + METHOD + "_luxry/feature_vectors.pt")
test_vectors = torch.load(DIR + "/" + METHOD + "_downstream/feature_vectors.pt")

# Convert to NumPy for subsequent operations
train_vectors = train_vectors.numpy()
train_vectors = train_vectors.T

test_vectors = test_vectors.numpy()
test_vectors = test_vectors.T

# Covariance matrix calculation
# cov_matrix = np.cov(train_vectors)

# Singular Value Decomposition
U, S, Vh = np.linalg.svd(train_vectors)

# the columns of u are the eigenvectors
# Matrxi[row, column]

# variances = []
# for i in range(len(S)):
#     p = U[:, i] @ test_vectors
#     # p = np.matmul(U[:, i], test_vectors)
#     var = np.var(p)
#     variances.append(var)
# variances = np.array(variances)

projection = U.T @ test_vectors
# projection = Vh @ test_vectors
variances = np.var(projection, axis=1)


# Logarithm of singular values
if LOG:
    v = np.log(variances)
else:
    v = variances
# log_v = variances
# print("Number of singular values:", len(S))
# np.save(dir + "/variance_analysis.npy", variances)

# Plotting
plt.plot(np.arange(len(v)), v)
plt.xlabel("Index")
if LOG:
    plt.ylabel("Logarithm of Variance")
else:
    plt.ylabel("Variance")
# plt.title("Singular Value Spectrum (Logarithmic)")
plt.grid(axis="y")
plt.plot()
# plt.show()
plt.savefig(DIR + "/variance_analysis_v2.png")
plt.close()
