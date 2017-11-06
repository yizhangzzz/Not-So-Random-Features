import numpy as np
import torch
from torch.autograd import Variable

def preprocess(X):
	#return (X - X.mean(0)) / X.std(0)
	return X - X.mean(0)

def pred_clf(model, Feat):
	#print(Feat)
	feat = Variable(Feat, requires_grad=False)
	pred = model(feat)
	pred_label = 2*(pred.data > 0).float() - 1
	return pred.data, pred_label

def train_hinge_clf(Feat, Y, model, C=1, n_iter=10, learning_rate=1e-4):
	optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.)
	feat = Variable(Feat, requires_grad=False)
	y = Variable(Y, requires_grad=False)
	for t in range(n_iter):
		pred = model(feat)
		loss = C * (1 - pred.mul(y)).clamp(min=0).sum() + 0.5*model.weight.pow(2).sum()

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

	pred = model(feat)
	margin = (1 - pred.mul(y)).clamp(min=0)
	return model, margin.data

def adaptive_compute_lam(w, x1, x2, a1, a2):
	p = 1.* a1.size(0) / (a1.size(0) + a2.size(0))
	q = 1.* a2.size(0) / (a1.size(0) + a2.size(0))

	feat_x1 = x1.mm(w) 
	feat_x2 = x2.mm(w) 

	a1_expand = a1.expand_as(feat_x1)
	a2_expand = a2.expand_as(feat_x2)

	phi_P_cos = torch.cos(feat_x1).mul(a1_expand).mean(0).mul_(p)
	phi_P_sin = torch.sin(feat_x1).mul(a1_expand).mean(0).mul_(p)

	phi_Q_cos = torch.cos(feat_x2).mul(a2_expand).mean(0).mul_(q)
	phi_Q_sin = torch.sin(feat_x2).mul(a2_expand).mean(0).mul_(q)

	lam = (phi_P_cos - phi_Q_cos).pow(2) + (phi_P_sin - phi_Q_sin).pow(2)

	return lam


def ogd_langevin_gpu(X1, X2, alpha1, alpha2, W_init, w_steps=10, alpha_steps=10, alpha_stepsize=0.01, w_stepsize=0.1, var=0.01, reg=0.1, C=1, temp=1e2):
	dtype = torch.cuda.FloatTensor
	x1 = Variable(X1.type(dtype), requires_grad=False)
	x2 = Variable(X2.type(dtype), requires_grad=False)
	a1 = Variable(alpha1.type(dtype), requires_grad=False)
	a2 = Variable(alpha2.type(dtype), requires_grad=False)

	p = 1.* a1.size(0) / (a1.size(0) + a2.size(0))
	q = 1.* a2.size(0) / (a1.size(0) + a2.size(0))


	def compute_loss(w, x1, x2, a1, a2):
		lam = adaptive_compute_lam(w, x1, x2, a1, a2)
		loss = -lam.mean() + reg*w.pow(2).mean() + 2*(a1.sum() + a2.sum()).div_((a1.size(0) + a2.size(0))**2)

		return loss, lam

	def compute_w_grad(w, x1, x2, a1, a2):
		w.requires_grad = True
		a1.requires_grad = False
		a2.requires_grad = False

		loss, lam = compute_loss(w, x1, x2, a1, a2)

		if w.grad is not None:
			w.grad.data.zero_()
		loss.backward()

		return loss.data, lam.data, w.grad.data

	def compute_alpha_grad(w, x1, x2, a1, a2):
		w.requires_grad = False 
		a1.requires_grad = True 
		a2.requires_grad = True 

		loss, _ = compute_loss(w, x1, x2, a1, a2)

		if a1.grad is not None:
			a1.grad.data.zero_()
		if a2.grad is not None:
			a2.grad.data.zero_()
		loss.backward()
		return loss.data, a1.grad.data, a2.grad.data

	def project_alpha(a1, a2, C, n_iter=10):
		for i in range(n_iter):
			a1.data.clamp_(0, C)
			a2.data.clamp_(0, C)
			decrement = (a1.data.sum() - a2.data.sum()) / (a1.data.size()[0] + a2.data.size()[0])
			a1.data -= decrement
			a2.data += decrement

		return a1.data, a2.data

	def accept_w(lam, w, w_grad, temp):
		#w_data_new = w.data - w_stepsize * (w_grad + torch.randn(w.size()).type(dtype) * var)
		w_data_new = w.data - w_stepsize * (w_grad + torch.randn(w.size()).type(dtype) * var)
		lam_new = adaptive_compute_lam(w_data_new, X1, X2, alpha1, alpha2)
		#acc_prob = torch.clamp((temp * (lam_new - lam)).exp_(), max=1.)
		#update_mask = torch.bernoulli(acc_prob).byte()
		#expanded_mask = update_mask.expand_as(w)
		#w.data.masked_scatter_(expanded_mask, w_data_new.masked_select(expanded_mask))
		#lam.masked_scatter_(update_mask, lam_new.masked_select(update_mask))

		return w.data, lam_new

	lam_max = 0
	W_max = 0
	w = Variable(W_init, requires_grad=False)
	for j in range(w_steps): 
		loss, lam, w_grad = compute_w_grad(w, x1, x2, a1, a2)

		lam_max_cand, idx = torch.max(lam, 0)
		if lam_max < lam_max_cand[0]:
			lam_max = lam_max_cand[0]
			W_max = w.data[:, idx]

		w.data = w.data - w_stepsize * (w_grad + torch.randn(w.size()).type(dtype) * var)

	w_last = Variable(W_max, requires_grad=False)
	for j in range(alpha_steps):
		l, a1_grad, a2_grad = compute_alpha_grad(w_last, x1, x2, a1, a2)
		a1.data += alpha_stepsize * a1_grad
		a2.data += alpha_stepsize * a2_grad
		a1.data, a2.data = project_alpha(a1, a2, C)

	return W_max, w.data, a1.data, a2.data, lam_max


def rbf_feat_gpu(X,W,B):
	F = X.mm(W) + B.expand(X.size(0), W.size(1))
	return np.sqrt(2./W.size(1))*torch.cos(F)

def weighted_rbf_feat_gpu(X,W,B, lam):
	F = X.mm(W) + B.expand(X.size(0), W.size(1))
	return np.sqrt(2.)*(lam / lam.norm()).sqrt().expand_as(F)*torch.cos(F)

