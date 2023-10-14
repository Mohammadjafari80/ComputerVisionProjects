import cv2
import numpy as np


def homography(points_s, points_d):
    A = []
    for point_src, point_target in zip(points_s, points_d):
        x_src, y_src = point_src
        x_target, y_target = point_target
        A.append([-x_src, -y_src, -1, 0, 0, 0, x_src * x_target, y_src * x_target, x_target])
        A.append([0, 0, 0, -x_src, -y_src, -1, x_src * y_target, y_src * y_target, y_target])

    A = np.array(A)
    u, s, vh = np.linalg.svd(A)
    return vh[-1].reshape((3, 3)) / vh[-1, -1]


logo = cv2.imread('./logo.png')
f = 500
p = 256
p_x = 1600
p_y = 900
R = np.array([[25, 0, 40], [0, 1, 0], [-40, 0, 25]], dtype=np.float32)
C = np.array([[-40, 0, 0]]).T

for i in range(3):
    R[i] = R[i] / np.linalg.norm(R[i])

I_C = np.hstack([np.eye(3), C])

r = 1
x_s = np.array([[1, -r, 25, 1],
                [2, r, 25, 1],
                [r, 2, 25, 1],
                [-r, 1, 25, 1]])

Rt = R @ I_C
Rt = np.vstack([Rt, np.array([0, 0, 0, 1])])
K_1 = np.array([[f, 0, p // 2], [0, f, p // 2], [0, 0, 1]])
K_1 = np.hstack([K_1, np.array([[0, 0, 0]]).T])
K_2 = np.array([[f, 0, p_x // 2], [0, f, p_y // 2], [0, 0, 1]])
K_2 = np.hstack([K_2, np.array([[0, 0, 0]]).T])

points_source = []
points_target = []

for i in range(4):
    x_w = x_s[i]
    x_c_1 = x_w
    x_c_2 = Rt @ x_w
    x_p_1 = K_2 @ x_c_1
    x_p_2 = K_1 @ x_c_2
    points_source.append((x_p_2 / x_p_2[-1])[:2][::-1])
    points_target.append((x_p_1 / x_p_1[-1])[:2][::-1])


H, mask = cv2.findHomography(np.array(points_source), np.array(points_target))
print('H is:')
print(H)
h, w, _ = logo.shape
pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
dst = cv2.perspectiveTransform(pts, H)
perspective_logo = cv2.warpPerspective(logo, H, (p_y, p_x))
cv2.imwrite('res12.jpg', perspective_logo)
