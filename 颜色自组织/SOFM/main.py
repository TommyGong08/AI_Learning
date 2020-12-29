import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from som_cm.io_util.image import loadRGB
from som_cm.core.hist_3d import Hist3D
from som_cm.core.som import SOMParam, SOM, SOMPlot

image_file = "test.png"
# Load image.
image = loadRGB(image_file)

# Color samples from 3D color histograms.
hist3D = Hist3D(image, num_bins=16)
color_samples = hist3D.colorCoordinates()

# Generate random data samples from color samples.
random_seed=100
num_samples=1000
random_ids = np.random.randint(len(color_samples) - 1, size=num_samples)
samples = color_samples[random_ids]

# 2D SOM: 32 x 32 map size.
param2D = SOMParam(h=32, dimension=2)
som2D = SOM(samples, param2D)

# Compute training process.
som2D.trainAll()

# SOM plotter.
som2D_plot = SOMPlot(som2D)

fig = plt.figure()

# Plot image.
fig.add_subplot(131)
plt.imshow(image)
plt.axis('off')

# Plot 2D SOM.
fig.add_subplot(132)
som2D_plot.updateImage()
plt.axis('off')

# Plot 2D SOM result in 3D RGB color space.
ax = fig.add_subplot(133, projection='3d')
som2D_plot.plot3D(ax)

plt.show()