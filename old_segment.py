# Olf filles for image segmentation

# Importing Necessary Libraries
# Displaying the sample image - Monochrome Format
from skimage import data
from skimage import filters
from skimage.color import rgb2gray
import matplotlib.pyplot as plt

from PIL import Image
import numpy as np
import scipy.misc
 
# Sample Image of scikit-image package
coffee = data.coffee()
gray_coffee = rgb2gray(coffee)
print (type(gray_coffee))

img = rgb2gray(coffee)*255
print (type(img))
print (img)

img = np.array(Image.fromarray(coffee).convert('L'))
print (type(img))
print (img)


# Setting the plot size to 15,15
"""
plt.figure(figsize=(15, 15))

for i in range(10):
	 
	# Iterating different thresholds
	#print( (gray_coffee > i*0.1) ) 
	binarized_gray = (gray_coffee > i*0.1)*1
	
	#print (type(binarized_gray))
	plt.subplot(5,2,i+1)
		
	# Rounding of the threshol value to 1 decimal point
	plt.title("Threshold: >"+str(round(i*0.1,1)))
	plt.imshow(binarized_gray, cmap = 'gray')
	 
plt.tight_layout()

plt.show()
"""