import numpy as np

data = np.load("/Users/nhoang2/Dropbox/Classes/IVIM_project/IVIM_deep_adaptive/FLAIR_prior/training_dictionary.npz")

# Pick any 1D array to check length — all should be same length
key = list(data.keys())[0]
num_voxels = data[key].shape[0]

print(f"Number of voxels/data points: {num_voxels}")
