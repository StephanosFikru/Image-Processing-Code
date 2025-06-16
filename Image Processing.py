import cv2
import numpy as np
from skimage import measure

image_path = "image.png"
image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError(f"Image file not found at {image_path}")

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

_, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

labels = measure.label(binary_image, connectivity=2)
properties = measure.regionprops(labels)

print(f"Number of detected properties: {len(properties)}")

min_cell_size = 5 

cell_count = 0
for prop in properties:
    if prop.area >= min_cell_size: 
        
        minr, minc, maxr, maxc = prop.bbox
        
        cell_image = image[minr:maxr, minc:maxc]
        
        cv2.imwrite(f"cell_{cell_count}.png", cell_image)
        cell_count += 1

print(f"Extracted {cell_count} cells.")
