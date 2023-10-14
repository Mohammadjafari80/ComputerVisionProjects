import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.feature import peak_local_max

img_1 = cv2.imread('./im01.jpg')
img_2 = cv2.imread('./im02.jpg')


def normalize(img):
    return cv2.normalize(img, np.zeros(img.shape), 0, 255, cv2.NORM_MINMAX)


def get_sobel_x(img):
    sobel_x = cv2.Sobel(src=img, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=3)
    return np.amax(np.abs(sobel_x), axis=-1)


def get_sobel_y(img):
    sobel_y = cv2.Sobel(src=img, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=3)
    return np.amax(np.abs(sobel_y), axis=-1)


def get_sobel(img):
    return get_sobel_x(img), get_sobel_y(img)


def get_gradient_magnitude(img):
    I_x, I_y = get_sobel(img)
    return np.sqrt(I_x ** 2 + I_y ** 2)


def gaussian_mask(size, sigma):
    center = size // 2
    K = np.zeros(shape=(size, size))
    for i in range(size):
        for j in range(size):
            K[i, j] = np.exp(-((i - center) ** 2 + (j - center) ** 2) / (2 * sigma ** 2)) / (2 * np.pi * sigma ** 2)

    return K / np.sum(K)


def non_maximum_suppression(img, n=20):
    return peak_local_max(img, min_distance=n)


def corner_harris(img, sigma, kernel_size, windows_size=15, k=0.05,
                  names=['res03_score', 'res05_thresh', 'res07_harris']):
    I_x, I_y = get_sobel(img)
    S_xy = cv2.GaussianBlur(I_x * I_y, dst=np.zeros_like(I_x), sigmaX=sigma, ksize=(kernel_size, kernel_size))
    S_xx = cv2.GaussianBlur(I_x * I_x, dst=np.zeros_like(I_x), sigmaX=sigma, ksize=(kernel_size, kernel_size))
    S_yy = cv2.GaussianBlur(I_y * I_y, dst=np.zeros_like(I_x), sigmaX=sigma, ksize=(kernel_size, kernel_size))

    W = gaussian_mask(2 * windows_size + 1, 5)

    structure_tensor = []

    for i in range(windows_size, img.shape[0] - windows_size):
        for j in range(windows_size, img.shape[1] - windows_size):
            window_xx = W * S_xx[i - windows_size: i + windows_size + 1, j - windows_size: j + windows_size + 1]
            window_yy = W * S_yy[i - windows_size: i + windows_size + 1, j - windows_size: j + windows_size + 1]
            window_xy = W * S_xy[i - windows_size: i + windows_size + 1, j - windows_size: j + windows_size + 1]

            structure_tensor.append((i, j, np.array([[np.sum(window_xx), np.sum(window_xy)],
                                                     [np.sum(window_xy), np.sum(window_yy)]])))

    R_img = np.zeros_like(I_x)

    for tensor in structure_tensor:
        i, j, M = tensor
        R_img[i, j] = np.linalg.det(M) - k * np.trace(M) ** 2

    cv2.imwrite(f'{names[0]}.jpg', normalize(R_img))
    R_max = np.max(R_img)
    thresh_img = np.where(R_img > R_max * 0.004, R_img, 0)
    cv2.imwrite(f'{names[1]}.jpg', normalize(thresh_img))

    corners = non_maximum_suppression(thresh_img, 18)

    img_copy = img.copy()
    for row, col in corners:
        img_copy = cv2.circle(img_copy, (col, row), 10, (50, 0, 255), -1)

    cv2.imwrite(f'{names[2]}.jpg', img_copy)

    return corners


def get_feature_vector(patch, div=4):
    h, w, _ = patch.shape
    h_div, w_div = h // div, w // div
    feature_vector = np.array([0])
    for i in range(div):
        for j in range(div):
            small_patch = patch[i * h_div: (i + 1) * h_div, j * w_div: (j + 1) * w_div]
            feature_vector = np.concatenate([feature_vector, small_patch.flatten()])

    return feature_vector


def match_features(corners_1, img_1, corners_2, img_2, n=200):
    row, col = corners_1.shape[0], corners_2.shape[0]
    scores = np.zeros(shape=(int(row), int(col)))
    first_to_second = []
    second_to_first = []
    pairs = []

    for i_index, coordinates1 in enumerate(corners_1):
        x1, y1 = coordinates1
        feature_i = get_feature_vector(img_1[x1 - n // 2: x1 + n // 2 + 1, y1 - n // 2: y1 + n // 2 + 1])
        for j_index, coordinates2 in enumerate(corners_2):
            x2, y2 = coordinates2
            feature_j = get_feature_vector(img_2[x2 - n // 2: x2 + n // 2 + 1, y2 - n // 2: y2 + n // 2 + 1])
            scores[i_index, j_index] = np.linalg.norm(feature_i - feature_j)

    threshold = 0.96

    for i in range(row):
        sorted = np.argsort(scores[i])
        ratio = scores[i][sorted[0]] / scores[i][sorted[1]]
        if ratio < threshold:
            first_to_second.append(sorted[0])
        else:
            first_to_second.append(-1)

    for i in range(col):
        sorted = np.argsort(scores[:, i])
        ratio = scores[sorted[0], i] / scores[sorted[1], i]
        if ratio < threshold:
            second_to_first.append(sorted[0])
        else:
            second_to_first.append(-1)

    for i in range(row):
        for j in range(col):
            if first_to_second[i] == j and second_to_first[j] == i:
                pairs.append((i, j))

    concat = cv2.hconcat([img_1, img_2])

    for pair in pairs:
        i, j = pair
        x1, y1 = corners_1[i]
        x2, y2 = corners_2[j]
        b, g, r = np.random.choice(np.arange(256), 3)
        color = (int(b), int(g), int(r))
        concat = cv2.line(concat, (int(y1), int(x1)), (int(y2) + img_1.shape[1], int(x2)), color=color, thickness=3)
        concat = cv2.circle(concat, (int(y1), int(x1)), radius=10, thickness=-1, color=color)
        concat = cv2.circle(concat, (int(y2) + img_1.shape[1], int(x2)), radius=10, thickness=-1, color=color)
        img_1 = cv2.circle(img_1, (int(y1), int(x1)), radius=10, thickness=-1, color=(50, 0, 255))
        img_2 = cv2.circle(img_2, (int(y2), int(x2)), radius=10, thickness=-1, color=(50, 0, 255))

    print(f'Number of corresponded pairs: {len(pairs)}')
    cv2.imwrite('res09_corres.jpg', img_1)
    cv2.imwrite('res10_corres.jpg', img_2)
    cv2.imwrite('res11.jpg', concat)
    return scores


img_1_gradient = get_gradient_magnitude(img_1)
img_2_gradient = get_gradient_magnitude(img_2)

cv2.imwrite('res01_grad.jpg', normalize(img_1_gradient))
cv2.imwrite('res02_grad.jpg', normalize(img_2_gradient))

img1_corners = corner_harris(img_1, 3, 11, names=['res03_score', 'res05_thresh', 'res07_harris'])
img2_corners = corner_harris(img_2, 3, 11, names=['res04_score', 'res06_thresh', 'res08_harris'])

match_features(img1_corners, img_1, img2_corners, img_2)
