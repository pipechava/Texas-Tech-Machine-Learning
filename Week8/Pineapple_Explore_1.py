# Introduction to Artificial Intelligence
# Exploration of pixel colors in pineapple image, part 1
# By Juan Carlos Rojas

import matplotlib.pyplot as plt
import skimage.io
import mpl_toolkits.mplot3d
import numpy as np

# Load image
img = skimage.io.imread("pineapple1.png")

# Display image
plt.imshow(img)
#plt.show()

# Reshape as a vector of pixels, each with 3 color components
pixels = img[:,:,0:3].reshape(-1,3)

# Pick a random subset
subset_len = 5000
total_len = pixels.shape[0]
reorder_map = np.random.permutation(total_len)
subset_map = reorder_map[0:subset_len]
pixels_subset = pixels[subset_map]

# Make a scatterplot of pixel colors on a 3D axis (R,G,B)
print("Plotting 3D scatterplot")
fig = plt.figure()
ax = mpl_toolkits.mplot3d.Axes3D(fig)
ax.scatter(pixels_subset[:,0], pixels_subset[:,1], pixels_subset[:,2], marker='.')
plt.show()

