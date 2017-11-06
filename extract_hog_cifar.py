from util import unpickle, load_cifar_data
import matplotlib.pyplot as plt
import numpy as np
import pickle
from skimage.feature import hog
from skimage import data, color, exposure


data = load_cifar_data('data/cifar-10-python/')
trainx = data['X_train']
trainy = data['Y_train']
testx = data['X_test']
testy = data['Y_test']

train_feat = []
test_feat = []

for i in range(trainx.shape[0]):
	if i % 10000 == 9999:
		print(i)
	image = color.rgb2gray(trainx[i,:,:,:])
	fd = hog(image, orientations=8, pixels_per_cell=(4, 4),
						cells_per_block=(1, 1), visualise=False)
	train_feat.append(fd.reshape(1, fd.shape[0]))

for i in range(testx.shape[0]):
	if i % 10000 == 9999:
		print(i)
	image = color.rgb2gray(testx[i,:,:,:])
	fd = hog(image, orientations=8, pixels_per_cell=(4, 4),
						cells_per_block=(1, 1), visualise=False)
	test_feat.append(fd.reshape(1, fd.shape[0]))

train_feat = np.concatenate(train_feat, axis=0)
test_feat = np.concatenate(test_feat, axis=0)

with open('data/cifar-10-python/cifar_hog.pkl', 'wb') as output:
	pickle.dump(dict(
		X_train=train_feat.astype(np.float32),
		Y_train=trainy.astype('int32'),
		X_test=test_feat.astype(np.float32),
		Y_test=testy.astype('int32')), output)