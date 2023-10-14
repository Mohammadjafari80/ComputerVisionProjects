#%%
import os
import cv2
from matplotlib.pyplot import axis
import numpy as np
import glob
from skimage.feature import hog
from sklearn import svm
from matplotlib import pyplot as plt
from sklearn.preprocessing import scale
from sklearn import metrics
import seaborn as sns
import pickle

#%%
LOAD = False
SAVE = True
load_path = './finalized_model.sav'
save_path = './model.sav'

#%%
img_addr = './Persepolis.jpg'
name, overlap_threshold, confidence_threshold = "res5.jpg", 0.15, 1.57
patch_sizes = np.linspace(80, 180, 5).astype(np.uint8)

#%%
def crop(img, margin=61):
    return img[margin:-margin, margin:-margin]

#%%
current_path = os.path.abspath(os.curdir)

#%%
### Positive
dataset_path = os.path.join(current_path, 'lfw')
print(dataset_path)
dataset_images = []

for i, data_path in enumerate(glob.glob(dataset_path + '*/*/*.jpg')):
    dataset_images.append(glob.glob(data_path)[0])

print(len(dataset_images))

#%%
### Negative
negative_dataset_path = os.path.join(current_path, '256_ObjectCategories')

negative_dataset_images = []

for i, data_path in enumerate(glob.glob(negative_dataset_path + '*/*/*.jpg')):
    addr = glob.glob(data_path)[0]
    if 'face' not in addr and 'people' not in addr:
        negative_dataset_images.append(glob.glob(data_path)[0])

print(len(negative_dataset_images))

#%%
TRAIN_COUNT = 10000
VALIDATION_COUNT = 1000
TEST_COUNT = 1000


#%%
def split(dataset_images, TRAIN_COUNT, VALIDATION_COUNT, TEST_COUNT):
    indices = np.random.choice(len(dataset_images), TRAIN_COUNT+VALIDATION_COUNT+TEST_COUNT, replace=False)
    selected = [dataset_images[i] for i in indices]
    train_set, val_set, test_set = selected[:TRAIN_COUNT], selected[TRAIN_COUNT: TRAIN_COUNT+VALIDATION_COUNT], selected[TRAIN_COUNT+VALIDATION_COUNT: TRAIN_COUNT+VALIDATION_COUNT+TEST_COUNT]
    return train_set, val_set, test_set

#%%
p_train_set, p_val_set, p_test_set = split(dataset_images, TRAIN_COUNT, VALIDATION_COUNT, TEST_COUNT)
n_train_set, n_val_set, n_test_set = split(negative_dataset_images, TRAIN_COUNT, VALIDATION_COUNT, TEST_COUNT)

#%%
WINDOW_SIZE = 64

#%%
p_val_images = [cv2.resize(crop(cv2.imread(img_addr)), (WINDOW_SIZE, WINDOW_SIZE)) for img_addr in p_val_set]
p_train_images = [cv2.resize(crop(cv2.imread(img_addr)), (WINDOW_SIZE, WINDOW_SIZE)) for img_addr in p_train_set]
p_test_images = [cv2.resize(crop(cv2.imread(img_addr)), (WINDOW_SIZE, WINDOW_SIZE)) for img_addr in p_test_set]
#%%
n_val_images = [cv2.resize(cv2.imread(img_addr), (WINDOW_SIZE, WINDOW_SIZE)) for img_addr in n_val_set]
n_train_images = [cv2.resize(cv2.imread(img_addr), (WINDOW_SIZE, WINDOW_SIZE)) for img_addr in n_train_set]
n_test_images = [cv2.resize(cv2.imread(img_addr), (WINDOW_SIZE, WINDOW_SIZE)) for img_addr in n_test_set]
#%%
orientations=9
pixels_per_cell=(4, 4)
cells_per_block=(2, 2)

#%%
def extract_features(p_images, n_images):
    features = []
    labels = []

    for img in n_images:
        feature_vector = hog(img, orientations=orientations, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block, multichannel=True)
        features.append(feature_vector)
        labels.append(0)

    for img in p_images:
        feature_vector = hog(img, orientations=orientations, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block, multichannel=True)
        features.append(feature_vector)
        labels.append(1)

    features = np.array(features)
    labels = np.array(labels)

    return features, labels


#%%
train_features , train_labels = None, None

if not LOAD:
    train_features , train_labels = extract_features(p_train_images, n_train_images)
val_features , val_labels = extract_features(p_val_images, n_val_images)
test_features , test_labels = extract_features(p_test_images, n_test_images)

#%%
c, gamma, kernel = 1, 'scale', 'rbf'

clf = None

if LOAD:
    clf = pickle.load(open(load_path, 'rb'))
else:
    clf = svm.SVC(C=c, gamma=gamma, kernel=kernel)
    clf.fit(scale(train_features), train_labels)
if SAVE:
    pickle.dump(clf, open(save_path, 'wb'))


#%%
preds = clf.predict(scale(val_features))
accuracy = metrics.accuracy_score(val_labels, preds)
print(f'Accuaracy is : {accuracy}')

#%%
test_scores = clf.decision_function(scale(test_features))
auc_score = metrics.roc_auc_score(test_labels, test_scores)
print(f'AUC score is : {auc_score}')
ap_score = metrics.average_precision_score(test_labels, test_scores)
print(f'AP score is : {ap_score}')

sns.set_theme()

#%%
fig, ax = plt.subplots(figsize=(7, 8))
metrics.plot_roc_curve(clf, scale(test_features), test_labels, ax=ax)
ax.set_xlim(-0.1, 1.1)
ax.set_ylim(-0.1, 1.1)
fig.savefig('res1.jpg', dpi=fig.dpi)

#%%
fig, ax = plt.subplots(figsize=(7, 8))
metrics.plot_precision_recall_curve(clf, scale(test_features), test_labels, ax=ax)
ax.set_xlim(-0.1, 1.1)
ax.set_ylim(-0.1, 1.1)
fig.savefig('res2.jpg', dpi=fig.dpi)


#%%
def detect_multi_scale(img, model, patch_sizes=np.linspace(80, 180, 5).astype(np.uint8)):
    coordinates = []
    scores = []
    for ps in patch_sizes:
        coordinates_c, scores_c = detect_faces(img, model, patch_size=ps)
        coordinates.extend(coordinates_c)
        scores.extend(scores_c)

    return coordinates, scores

def detect_faces(img, model, patch_size):
    step = 10
    rows, cols, channels = img.shape
    new_img = np.zeros((rows + patch_size, cols + patch_size, 3))
    new_img[:rows, :cols] = img
    coordinates = []
    patch_feature_vectors = []
    for r in range(0, rows, step):
        for c in range(0, cols, step):
            patch = new_img[r:r+patch_size, c:c+patch_size]
            patch = cv2.resize(patch, (WINDOW_SIZE, WINDOW_SIZE))
            feature_vector = hog(patch, orientations=orientations, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block, multichannel=True)
            patch_feature_vectors.append(feature_vector)
            coordinates.append((r,c, patch_size))
            
    scores = model.decision_function(scale(np.array(patch_feature_vectors)))

    return coordinates, scores


def non_max_suppression(boxes, overlapThresh, score):
	pick = []
	x1 = boxes[:,0]
	y1 = boxes[:,1]
	x2 = boxes[:,2]
	y2 = boxes[:,3]

	# compute the area of the bounding boxes and sort the bounding
	# boxes by their score
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = np.argsort(score)

	while len(idxs) > 0:
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)
		xx1 = np.maximum(x1[i], x1[idxs[:last]])
		yy1 = np.maximum(y1[i], y1[idxs[:last]])
		xx2 = np.minimum(x2[i], x2[idxs[:last]])
		yy2 = np.minimum(y2[i], y2[idxs[:last]])
		w = np.maximum(0, xx2 - xx1 + 1)
		h = np.maximum(0, yy2 - yy1 + 1)
		overlap = (w * h) / area[idxs[:last]]
		idxs = np.delete(idxs, np.concatenate(([last],
			np.where(overlap > overlapThresh)[0])))
	return boxes[pick].astype("int")

def identify(img, patch_sizes=np.linspace(80, 180, 5).astype(np.uint8)):
    coordinates, scores = detect_multi_scale(img, clf, patch_sizes=patch_sizes)
    args = np.argsort(np.array(scores))[::-1]
    scores = np.array(scores)[args]
    coordinates = np.array(coordinates)[args]
    boxes = []
    for r, c, length in coordinates:
        start = (c, r)
        end = (c+length, r+length)
        boxes.append((start[0], start[1], end[0], end[1]))
    return boxes, scores, coordinates


# %%
img = cv2.imread(img_addr)

#%%
bounding_boxes, s, c = identify(img, patch_sizes)

# %%
new_img = img.copy()
new_boxes = non_max_suppression(np.array(bounding_boxes)[s > confidence_threshold], overlap_threshold, s[s > confidence_threshold])
for x1, y1, x2, y2 in new_boxes:
    start = (x1, y1)
    end = (x2, y2)
    B, G, R = np.random.choice(255, 3).tolist()
    cv2.rectangle(new_img, start, end, (int(B), int(G), int(R)), 3)
        
cv2.imwrite(name, new_img)
# %%
