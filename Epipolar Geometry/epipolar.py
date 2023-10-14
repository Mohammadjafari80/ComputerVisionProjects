import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy.linalg import null_space

def concat_h(img1, img2):
    img2 = img2.copy()
    if img2.shape[0] > img1.shape[0]:
        ratio = img1.shape[0] / img2.shape[0]
        img2 = cv2.resize(img2, (int(img2.shape[1] * ratio), img1.shape[0]))
    black_pixels = np.zeros(shape=(img1.shape[0], img2.shape[1], 3)).astype(np.uint8)
    black_pixels[: img2.shape[0], :] = img2
    return cv2.hconcat([img1, black_pixels])

img1 = cv2.imread('01.JPG') 
img2 = cv2.imread('02.JPG')
sift = cv2.SIFT_create()


# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)


# FLANN parameters

bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2, k=2)
pts1 = []
pts2 = []

# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches):
    if m.distance < 0.6 *n.distance:
        pts2.append(kp2[m.trainIdx].pt)
        pts1.append(kp1[m.queryIdx].pt)


pts1 = np.int32(pts1)
pts2 = np.int32(pts2)
F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_LMEDS)


# We select only inlier points
inliers1 = pts1[mask.ravel()==1]
inliers2 = pts2[mask.ravel()==1]
outliers1 = pts1[mask.ravel()==0]
outliers2 = pts2[mask.ravel()==0]



radius = 10
correspondences_1 = img1.copy()
for pts in inliers1:
    x_1, y_1 = pts
    correspondences_1 = cv2.circle(correspondences_1, (int(x_1), int(y_1)), radius=radius, thickness=-1, color=(0, 255, 0))

for pts in outliers1:
    x_1, y_1 = pts
    correspondences_1 = cv2.circle(correspondences_1, (int(x_1), int(y_1)), radius=radius, thickness=-1, color=(0, 0, 255))

correspondences_2 = img2.copy()
for pts in inliers2:
    x_1, y_1 = pts
    correspondences_2 = cv2.circle(correspondences_2, (int(x_1), int(y_1)), radius=radius, thickness=-1, color=(0, 255, 0))

for pts in outliers2:
    x_1, y_1 = pts
    correspondences_2 = cv2.circle(correspondences_2, (int(x_1), int(y_1)), radius=radius, thickness=-1, color=(0, 0, 255))

correspondences = concat_h(correspondences_1, correspondences_2)
cv2.imwrite('res05.jpg', correspondences)


b = np.zeros(shape=(3, 1))

u, s, vh = np.linalg.svd(F)
e1 = vh[-1]
u, s, vh = np.linalg.svd(F.T)
e2 = vh[-1]

e1 /= e1[-1]
e2 /= e2[-1]

print(F)

print(e1)
print(e2)


import seaborn as sns
sns.set_theme()

img = plt.imread('01.JPG')
fig = plt.figure(figsize=(40, 40))
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
ax.imshow(img)
ax.plot(int(e1[0]),  int(e1[1]), marker='o', color='mediumvioletred' , markersize=10)
plt.grid(True)
plt.savefig('res06.jpg', bbox_inches ="tight", pad_inches = 1,)

img = plt.imread('02.JPG')
fig = plt.figure(figsize=(40, 40))
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
ax.imshow(img)
ax.plot(int(e2[0]),  int(e2[1]), marker='o', color='mediumvioletred', markersize=10)
plt.grid(True)
plt.savefig('res07.jpg', bbox_inches ="tight", pad_inches = 1,)



def draw_line(pts1, pts2, img, color):
    x_1, y_1 = pts1
    y_2, x_2 = pts2
    cv2.circle(img, (x_1, y_1), thickness=-1, radius=2 * radius, color=color)
    a = (y_1 - y_2) / (x_1 - x_2)
    b = y_1 - a * x_1
    x_s, x_t = -10000, 10000
    y_s, _y_t = int(a * x_s + b), int(a * x_t + b)
    cv2.line(img, (x_s, y_s), (x_t, _y_t), thickness=7, color=color)


random_indices = np.random.choice(len(inliers1), 15, replace=False)

lines_1 = img1.copy()
lines_2 = img2.copy()

for pts_1, pts_2 in zip(inliers1[random_indices], inliers2[random_indices]):
    b, g, r = np.random.choice(np.arange(256), 3)
    color = (int(b), int(g), int(r))
    draw_line(pts_1, e1[:2][::-1], lines_1, color=color), 
    draw_line(pts_2, e2[:2][::-1], lines_2, color=color)


lines = concat_h(lines_1, lines_2)
cv2.imwrite('res08.jpg', lines)
