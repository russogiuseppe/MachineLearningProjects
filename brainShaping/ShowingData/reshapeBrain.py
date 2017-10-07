import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import regionprops
import matplotlib.patches as mpatches
from skimage.morphology import label
from skimage.filter import threshold_adaptive
from skimage import filter
from skimage.feature import peak_local_max
from skimage.color import gray2rgb

#problem the image is not in RGB
import numpy as np

print "sono qua"
X_TrainSet = np.load('/home/giuseppe/ml-project/data/X_train.npy')
#improve this function
print "sono qua"
#we should find a way to select all of the features

brainCollection = np.reshape(X_TrainSet,(-1,176,208,176))

#image

youngBrain = brainCollection[2,37,:,:]
#image = gray2rgb(youngBrain)
print "sono qua"
#plot the original brain
fig, axes = plt.subplots(ncols=2, nrows=3,
                         figsize=(8, 4))
ax0, ax1, ax2, ax3, ax4, ax5  = axes.flat
ax0.imshow(youngBrain, cmap = "gray")
ax0.set_title('Original', fontsize=24)
ax0.axis('off')


print "sono qua"
#Histograms of the intesity values
values, bins = np.histogram(youngBrain,
                            bins=np.arange(256))

ax1.plot(bins[:-1], values, lw=2, c='k')
ax1.set_xlim(xmax=256)
ax1.set_yticks([0, 400])
ax1.set_aspect(.2)
ax1.set_title('Histogram', fontsize=24)
#background


background = threshold_adaptive(youngBrain, 95, offset=-15)

ax2.imshow(background, cmap=plt.cm.gray)
ax2.set_title('Adaptive threshold', fontsize=24)
ax2.axis('off')

#localMaxima
print "sono qua"

youngBrain = gray2rgb(youngBrain)
coordinates = peak_local_max(youngBrain, min_distance=20)

ax3.imshow(youngBrain, cmap=plt.cm.gray)
ax3.autoscale(False)
ax3.plot(coordinates[:, 1],
         coordinates[:, 0], c='r.')
ax3.set_title('Peak local maxima', fontsize=24)
ax3.axis('off')

print "sono qua"
edges = filter.canny(youngBrain, sigma=3,
                     low_threshold=10,
                     high_threshold=80)

ax4.imshow(edges, cmap=plt.cm.gray)
ax4.set_title('Edges', fontsize=24)
ax4.axis('off')


label_image = label(edges)

ax5.imshow(youngBrain, cmap=plt.cm.gray)
ax5.set_title('Labeled items', fontsize=24)
ax5.axis('off')

for region in regionprops(label_image):
    # Draw rectangle around segmented coins.
    minr, minc, maxr, maxc = region.bbox
    rect = mpatches.Rectangle((minc, minr),
                              maxc - minc,
                              maxr - minr,
                              fill=False,
                              edgecolor='red',
                              linewidth=2)
    ax5.add_patch(rect)

plt.tight_layout()
plt.show()