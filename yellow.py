import cv2
import numpy as np

# grid_img = cv2.imread('50000.jpg')
grid_img = cv2.imread('grid_img.png')

small_grid_img = cv2.pyrDown(grid_img)

img_b, img_g, img_r = cv2.split(small_grid_img)  
img_b = img_b.astype(np.float32)

print(img_b.shape, img_b.dtype)

img_b = img_b + 1
ratio_rb = img_r/img_b
ratio_gb = img_g/img_b

maxrb = np.max(ratio_rb)
maxgb = np.max(ratio_gb)
print(maxrb, maxgb)

minrb = np.min(ratio_rb)
mingb = np.min(ratio_gb)
print(minrb, mingb)

mask_rb = ratio_rb > 1.3
mask_gb = ratio_gb > 1.2
mask = mask_rb & mask_rb
print(ratio_rb)
# print(mask)
color = np.full(small_grid_img.shape, (0, 0, 0), dtype=np.uint8)
color[mask] = (255,255,255)

cv2.imwrite('yellow.jpg', color)
# cv2.imshow('grid_detection',color)
# key = cv2.waitKey(0)

# ratio_rb = ratio_rb *img_b/255
# maxrb = np.max(ratio_rb)
# print(maxrb)
# ratio_rb *= ratio_gb

ratio_rb = ratio_rb * 255 / maxrb

maxrb = np.max(ratio_rb)
print(maxrb)

img_rb = ratio_rb.astype(np.uint8)
cv2.imwrite('img_rb.jpg', img_rb)
cv2.imshow('img_rb',img_rb)
key = cv2.waitKey(0)

maxrb = np.max(img_rb)
print(maxrb)