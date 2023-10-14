import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.path as mplPath

img_1 = cv2.imread('./im03.jpg')
img_2 = cv2.imread('./im04.jpg')



def concat_h(img1, img2):
    img2 = img2.copy()
    if img2.shape[0] > img1.shape[0]:
        ratio = img1.shape[0] / img2.shape[0]
        img2 = cv2.resize(img2, (int(img2.shape[1] * ratio), img1.shape[0]))
    black_pixels = np.ones(shape=(img1.shape[0], img2.shape[1], 3)).astype(np.uint8) * 255
    black_pixels[: img2.shape[0], :] = img2
    return cv2.hconcat([img1, black_pixels])


def RANSAC(kp_1, kp_2, img1, img2, p=0.99):
    w = 0
    kp1, kp2 = kp_1, kp_2
    kp_1_i = [x.pt[::-1] for x in kp_1]
    kp_2_i = [x.pt[::-1] for x in kp_2]
    kp_1 = [x.pt for x in kp_1]
    kp_2 = [x.pt for x in kp_2]
    kp_1, kp_2 = np.array(kp_1), np.array(kp_2)
    kp_1_i, kp_2_i = np.array(kp_1_i), np.array(kp_2_i)
    keypoints_pairs = list(zip(kp_1, kp_2))
    N = float('inf')
    counter = 0
    t = 65
    H_matrices = []
    number_of_inliers = []
    total = kp_1.shape[0]

    while min(10000, N) > counter:
        samples = np.random.choice(np.arange(len(keypoints_pairs)), 4).astype(int)
        h = homography(kp_2[samples], kp_1[samples])
        inliers_count, _ = get_inliers(h, kp_2, kp_1, t)
        H_matrices.append(h)
        number_of_inliers.append(inliers_count)
        new_w = inliers_count / total

        if new_w > w:
            w = new_w
            N = int(np.log(1 - p) / np.log(1 - w ** 4))

        counter += 1
    print('Count is : ', counter)
    prev_H = H_matrices[np.argmax(np.array(number_of_inliers))]
    current_H = find_homography(prev_H, kp_2, kp_1, t)
    inliers_count, _ = get_inliers(current_H, kp_2, kp_1, t)
    H_matrices.append(current_H)
    number_of_inliers.append(inliers_count)

    for i in range(200):
        prev_H = current_H
        current_H = find_homography(prev_H, kp_2, kp_1, t)
        inliers_count, _ = get_inliers(current_H, kp_2, kp_1, t)
        H_matrices.append(current_H)
        number_of_inliers.append(inliers_count)

    H = H_matrices[np.argmax(np.array(number_of_inliers))]
    print('H is:')
    print(H)
    mask = get_mask(H, kp_2, kp_1, t)
    H_inv = np.linalg.inv(H)

    matchesMask = mask

    kp1_inliers, kp2_inliers = [], []
    kp1_outliers, kp2_outliers = [], []

    for i, in_or_out in enumerate(matchesMask):
        if matchesMask[i] == 1:
            kp1_inliers.append(kp1[i])
            kp2_inliers.append(kp2[i])
        else:
            kp1_outliers.append(kp1[i])
            kp2_outliers.append(kp2[i])

    concat_image = concat_h(img1, img2)

    for point_1, point_2 in zip(kp1_outliers, kp2_outliers):
        x_1, y_1 = point_1.pt
        x_2, y_2 = point_2.pt
        concat_image = cv2.line(concat_image, (int(x_1), int(y_1)), (int(x_2) + img1.shape[1], int(y_2)), (255, 0, 0),
                                2)
        concat_image = cv2.circle(concat_image, (int(x_1), int(y_1)), radius=6, thickness=-1, color=(255, 0, 0))
        concat_image = cv2.circle(concat_image, (int(x_2) + img1.shape[1], int(y_2)), radius=6, thickness=-1,
                                  color=(255, 0, 0))

    for point_1, point_2 in zip(kp1_inliers, kp2_inliers):
        x_1, y_1 = point_1.pt
        x_2, y_2 = point_2.pt
        concat_image = cv2.line(concat_image, (int(x_1), int(y_1)), (int(x_2) + img1.shape[1], int(y_2)), (0, 0, 255),
                                2)
        concat_image = cv2.circle(concat_image, (int(x_1), int(y_1)), radius=6, thickness=-1, color=(0, 0, 255))
        concat_image = cv2.circle(concat_image, (int(x_2) + img1.shape[1], int(y_2)), radius=6, thickness=-1,
                                  color=(0, 0, 255))

    cv2.imwrite('res26.jpg', concat_image)

    h, w, _ = img1.shape
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, H_inv)

    poly_path = mplPath.Path(dst.reshape((-1, 2)))
    has_missing = False
    concat_image = concat_h(img1, img2)
    for point_1, point_2 in zip(kp1_inliers[:30], kp2_inliers):
        x_1, y_1 = point_1.pt
        x_2, y_2 = point_2.pt
        if not poly_path.contains_point((x_2, y_2)):
            has_missing = True
            concat_image = cv2.line(concat_image, (int(x_1), int(y_1)), (int(x_2) + img1.shape[1], int(y_2)),
                                    (255, 255, 0),
                                    2)
            concat_image = cv2.circle(concat_image, (int(x_1), int(y_1)), radius=6, thickness=-1, color=(255, 255, 0))
            concat_image = cv2.circle(concat_image, (int(x_2) + img1.shape[1], int(y_2)), radius=6, thickness=-1,
                                      color=(255, 255, 0))

    if has_missing:
        cv2.imwrite('res18_missing.jpg', concat_image)

    poly_lines = np.zeros(shape=(img2.shape[0] + 200, img2.shape[1], 3))
    poly_lines[:img2.shape[0], :] = img2
    poly_lines = cv2.polylines(poly_lines, [np.int32(dst)], True, (255, 255, 255), 3, cv2.LINE_AA)
    poly_mask = cv2.fillPoly(np.zeros_like(poly_lines), [np.int32(dst)], (0, 255, 0))
    cv2.imwrite('res28.jpg', concat_h(img1, 0.15 * poly_mask + 0.85 * poly_lines))
    shift_x, shift_y = dst[0][0].astype(int)
    # shift_x, shift_y = 0, 0
    offset = 500

    H, mask = cv2.findHomography(dst , pts + 2 * np.array([shift_x + offset + 600, shift_y + offset]))
    w, h = img2.shape[:2][::-1]
    transformed_img_2 = cv2.warpPerspective(img2, H, (int(w * 4.3), int(h * 4.3)))
    cv2.imwrite('res29.jpg', transformed_img_2)

    transformed_img_2 = cv2.warpPerspective(0.15 * poly_mask + 0.85 * poly_lines, H, (int(w * 4.3), int(h * 4.3)))
    cv2.imwrite('res30.jpg', concat_h(img1, transformed_img_2))
    print('Number of inliers: ', len(kp1_inliers))
    
    h, w, _ = transformed_img_2.shape
    white_background = np.ones((h, w, 3), np.uint8) * 255

    # Step 2: Create a mask of the non-black pixels in transformed_img_2
    non_black_mask = np.all(transformed_img_2 != [0, 0, 0], axis=2)

    # Step 3: Blend the images: keep the non-black pixels of transformed_img_2, and the rest from white_background
    blended_img = transformed_img_2.copy()
    blended_img[~non_black_mask] = white_background[~non_black_mask]

    # Now blended_img has a white background instead of black
    cv2.imwrite('res30_with_white_background.jpg', concat_h(img1, blended_img))


def get_inliers(H, kp_src, kp_target, t):
    total = len(kp_src)
    inliers_count = 0
    inliers = []
    # H_inv = np.linalg.inv(H)
    for i in range(total):
        x_src, y_src = kp_src[i]
        x_target, y_target = kp_target[i]
        projection = H @ np.array([x_src, y_src, 1])
        projection = projection / (projection[-1] + 1e-8)
        x_estimate, y_estimate, _ = projection
        d = np.linalg.norm(np.array([x_target, y_target, 1]) - projection)
        if d < t:
            inliers_count += 1
            inliers.append((kp_src[i], kp_target[i]))

    return inliers_count, inliers


def get_mask(H, kp_src, kp_target, t):
    total = len(kp_src)
    mask = []
    # H_inv = np.linalg.inv(H)
    for i in range(total):
        x_src, y_src = kp_src[i]
        x_target, y_target = kp_target[i]
        projection = H @ np.array([x_src, y_src, 1])
        projection = projection / projection[-1]
        d = np.linalg.norm(np.array([x_target, y_target, 1]) - projection)
        if d < t:
            mask.append(1)
        else:
            mask.append(0)

    return mask


def find_homography(H, kp1, kp2, t):
    _, inliers = get_inliers(H, kp1, kp2, t)
    A = []
    for point_src, point_target in inliers:
        x_src, y_src = point_src
        x_target, y_target = point_target
        A.append([-x_src, -y_src, -1, 0, 0, 0, x_src * x_target, y_src * x_target, x_target])
        A.append([0, 0, 0, -x_src, -y_src, -1, x_src * y_target, y_src * y_target, y_target])

    A = np.array(A)
    u, s, vh = np.linalg.svd(A)
    return vh[-1].reshape((3, 3)) / vh[-1, -1]


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




sift = cv2.SIFT_create()

keypoints_1, descriptors_1 = sift.detectAndCompute(img_1, None)
keypoints_2, descriptors_2 = sift.detectAndCompute(img_2, None)

print('length of first image keypoints is : ', len(keypoints_1))
print('length of second image keypoints is : ', len(keypoints_2))

keypoints_image_1 = cv2.drawKeypoints(img_1, keypoints_1, np.zeros_like(img_1), color=(0, 255, 0),
                                      flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)

keypoints_image_2 = cv2.drawKeypoints(img_2, keypoints_2, np.zeros_like(img_1), color=(0, 255, 0),
                                      flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)

cv2.imwrite('res22_corners.jpg', concat_h(keypoints_image_1, keypoints_image_2))

# feature matching
bf = cv2.BFMatcher()


matches = bf.knnMatch(descriptors_1, descriptors_2, k=2)

# Apply ratio test
print('Number of matches before ratio test: ', len(matches))

adequate_matches = []
for m, n in matches:
    if m.distance < 0.8 * n.distance:
        adequate_matches.append([m])

matches = adequate_matches
print('Number of matches after ratio test: ', len(matches))
matched_keypoints_1 = []
matched_keypoints_2 = []
for match in matches:
    matched_keypoints_1.append(keypoints_1[match[0].queryIdx])
    matched_keypoints_2.append(keypoints_2[match[0].trainIdx])


img_copy_1 = keypoints_image_1.copy()
for match in matched_keypoints_1:
    x_1, y_1 = match.pt
    img_copy_1 = cv2.circle(img_copy_1, (int(x_1), int(y_1)), radius=7, thickness=-1, color=(255, 0, 0))

img_copy_2 = keypoints_image_2.copy()
for match in matched_keypoints_2:
    x_1, y_1 = match.pt
    img_copy_2 = cv2.circle(img_copy_2, (int(x_1), int(y_1)), radius=7, thickness=-1, color=(255, 0, 0))

cv2.imwrite('res23_correspondences.jpg', concat_h(img_copy_1, img_copy_2))

correspondences = concat_h(img_copy_1, img_copy_2)
for point_1, point_2 in zip(matched_keypoints_1, matched_keypoints_2):
    x_1, y_1 = point_1.pt
    x_2, y_2 = point_2.pt
    correspondences = cv2.line(correspondences, (int(x_1), int(y_1)), (int(x_2) + img_1.shape[1], int(y_2)),
                               (255, 0, 0),
                               2)
    correspondences = cv2.circle(correspondences, (int(x_1), int(y_1)), radius=7, thickness=-1, color=(255, 0, 0))
    correspondences = cv2.circle(correspondences, (int(x_2) + img_1.shape[1], int(y_2)), radius=7, thickness=-1,
                                 color=(255, 0, 0))

cv2.imwrite('res24_matches.jpg', correspondences)

random_matches = list(np.array(matches)[np.random.choice(np.arange(len(matches)), 20).astype(int)])

random_matched_keypoints_1 = []
random_matched_keypoints_2 = []

for match in random_matches:
    random_matched_keypoints_1.append(keypoints_1[match[0].queryIdx])
    random_matched_keypoints_2.append(keypoints_2[match[0].trainIdx])

correspondences_random = concat_h(img_1, img_2)

for point_1, point_2 in zip(random_matched_keypoints_1, random_matched_keypoints_2):
    x_1, y_1 = point_1.pt
    x_2, y_2 = point_2.pt
    correspondences_random = cv2.line(correspondences_random, (int(x_1), int(y_1)),
                                      (int(x_2) + img_1.shape[1], int(y_2)), (255, 0, 0),
                                      2)
    correspondences_random = cv2.circle(correspondences_random, (int(x_1), int(y_1)), radius=7, thickness=-1,
                                        color=(255, 0, 0))
    correspondences_random = cv2.circle(correspondences_random, (int(x_2) + img_1.shape[1], int(y_2)), radius=7,
                                        thickness=-1,
                                        color=(255, 0, 0))

cv2.imwrite('res25.jpg', correspondences_random)


RANSAC(matched_keypoints_1, matched_keypoints_2, img_1, img_2)
