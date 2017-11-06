import numpy as np
import pickle
from nsrf import *
import argparse

parser = argparse.ArgumentParser(description='classes')
parser.add_argument('--classP', metavar='classP', type=int, nargs='+')
parser.add_argument('--classQ', metavar='classQ', type=int, nargs='+')
args = parser.parse_args()
print(args.classP)
print(args.classQ)

dtype = torch.cuda.FloatTensor
C = 1

with open('data/cifar-10-python/cifar_hog.pkl', 'r') as infile:
	data = pickle.load(infile)

X_train_total = data['X_train']
Y_train_total = data['Y_train']
X_test_total = data['X_test']
Y_test_total = data['Y_test']

classP, classQ = args.classP[0], args.classQ[0]

cond_train = np.logical_or(Y_train_total==classP, Y_train_total==classQ)
X_train = X_train_total[cond_train,:]
Y_train = Y_train_total[cond_train]
Y_train[Y_train == classP] = 0
Y_train[Y_train == classQ] = 1

cond_test = np.logical_or(Y_test_total==classP,Y_test_total==classQ)
X_test = X_test_total[cond_test,:]
X_test = X_test_total[cond_test,:]
Y_test = Y_test_total[cond_test]
Y_test[Y_test == classP] = 0
Y_test[Y_test == classQ] = 1

X_train = preprocess(X_train)
X_test = preprocess(X_test)

n_samples, n_dim, n_features = X_train.shape[0], X_train.shape[1], 1000

X1 = X_train[Y_train==0,:]
X2 = X_train[Y_train==1,:]
Y_train = 2*Y_train - 1
Y_test = 2*Y_test - 1


X1_gpu = torch.from_numpy(X1).type(dtype)
X2_gpu = torch.from_numpy(X2).type(dtype)
X_train_gpu = torch.from_numpy(X_train).type(dtype)
X_test_gpu = torch.from_numpy(X_test).type(dtype)
Y_train_gpu = torch.from_numpy(Y_train.reshape(-1,1)).type(dtype)
Y_test_gpu = torch.from_numpy(Y_test.reshape(-1,1)).type(dtype)

print("Loading finished")


alpha1 = C/2.*torch.ones(X1.shape[0], 1).type(dtype)
alpha2 = C/2.*torch.ones(X2.shape[0], 1).type(dtype)

W_gpu = None

for i in range(5000):

	W_init = 0.3*torch.cuda.FloatTensor(n_dim, 500).normal_(0, 1)
	W_max, W_init, alpha1, alpha2, lam_max = ogd_langevin_gpu(X1_gpu, X2_gpu, alpha1, alpha2, W_init, w_steps=100, alpha_steps=1, alpha_stepsize=1e3, w_stepsize=1e4, var=1e-6, reg=0.01, C=C, temp=1e1)

	if i == 0:
		W_gpu = W_max
	else:
		W_gpu = torch.cat((W_gpu, W_max), 1)

	if i % 100 == 99:
		print("iteration: %d" % i)
		print((alpha1>0.95*C).type(dtype).mean())
		print((alpha1<0.05*C).type(dtype).mean())
		print((alpha2>0.95*C).type(dtype).mean())
		print((alpha2<0.05*C).type(dtype).mean())
		
		n_features = W_gpu.size(1)
		B_gpu = 2*np.pi*torch.rand(1, n_features).type(dtype)

		Feat_train_gpu = rbf_feat_gpu(X_train_gpu, W_gpu, B_gpu)
		Feat_test_gpu = rbf_feat_gpu(X_test_gpu, W_gpu, B_gpu)

		model = torch.nn.Linear(n_features, 1).cuda()
		model, margin = train_hinge_clf(Feat_train_gpu, Y_train_gpu, model, C=C, n_iter=10000, learning_rate=1e-3)

		_, Y_train_pred = pred_clf(model, Feat_train_gpu) 
		print("training accuracy: %.4f" % (Y_train_pred == Y_train_gpu).float().mean())

		_, Y_test_pred = pred_clf(model, Feat_test_gpu)
		print("testing accuracy: %.4f" % (Y_test_pred == Y_test_gpu).float().mean())


