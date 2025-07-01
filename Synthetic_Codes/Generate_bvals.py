import numpy as np

# Define base b-values
bvals_base = [0, 5, 7, 10, 15, 20, 30, 40, 50, 60, 100, 200, 400, 700, 1000]

# Repeat each b-value 3 times
bvals_repeated = np.repeat(bvals_base, 3)

# Output path
output_path = "/scratch/nhoang2/IVIM_NeuroCovid/Data/bvals.txt"

# Save as space-separated values (single row)
np.savetxt(output_path, bvals_repeated.reshape(1, -1), fmt='%d', delimiter=' ')

print(f"[SAVED] bvals.txt with shape {bvals_repeated.shape} → {output_path}")
