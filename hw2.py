import numpy as np
import matplotlib.pyplot as plt
import cv2

image_path = r"C:\ddd\data\svd_demo1.jpg"
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
A = img.astype(dtype=np.float64)
U, Sigma, VT = np.linalg.svd(A, full_matrices=False)
def compute_energy(X):
    return np.sum(X**2)
keep_r = 201
rs = np.arange(1, keep_r)
energy_A = compute_energy(A)
energy_N = np.zeros(keep_r)
snr = np.zeros(keep_r)
for r in rs:
    A_bar = (U[:, :r] * Sigma[:r]) @ VT[:r, :]
    Noise = A - A_bar
    energy_N[r] = compute_energy(Noise)
    snr[r] = 10 * np.log10((energy_A - energy_N[r]) / energy_N[r])
plt.figure(figsize=(8, 6))
plt.plot(rs, snr[1:keep_r])
plt.xlabel('r')
plt.ylabel('dB')
plt.grid()
plt.show()
r_values = [5, 20, 50, 100]
fig, axes = plt.subplots(1, len(r_values) + 1, figsize=(15, 5))
axes[0].imshow(A, cmap='gray')
axes[0].set_title("Original Image")
axes[0].axis('off')
for i, r in enumerate(r_values):
    A_bar = (U[:, :r] * Sigma[:r]) @ VT[:r, :]
    axes[i + 1].imshow(A_bar, cmap='gray')
    axes[i + 1].set_title(f"r = {r}")
    axes[i + 1].axis('off')
plt.tight_layout()
plt.show()
