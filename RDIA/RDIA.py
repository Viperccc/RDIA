import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

from grad_utils import grad_logloss_theta_lr
from grad_utils import batch_grad_logloss_lr
from inverse_hvp import inverse_hvp_lr_newtonCG
from dataset import load_data_v1,select_from_one_class,load_data
import argparse
import time
import pdb
import os
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type = str, default = 'cancer')
parser.add_argument('--noise', type = float, help = 'corruption rate, should be less than 1', default = 0.4)
parser.add_argument('--alpha', type = float, help = 'Hyperparameter to control relabeling numbers', default = 0.0002)


args = parser.parse_args()
dataset_name = args.dataset
#select the dataset used
#dataset_name = "mnist"
#dataset_name = "covtype"
#dataset_name = "a1a"
#dataset_name = "realsim"
#dataset_name = "avazu_app"
#dataset_name = "criteo"
#dataset_name = "diabetes"
#dataset_name = "news20"
# parameter for the sigmoid sampling function
sigmoid_k = 10
# regularization parameter for Logistic Regression
C = 0.1
# sample ratio
sample_ratio = 0.95
# flip ratio
flip_ratio = args.noise

start_time = time.time()
# load data, pick 30% as the Va set
x_train,y_train,x_va,y_va,x_te,y_te = load_data_v1(dataset_name,va_ratio=0.3)
#x_train,y_train,x_va,y_va,x_te,y_te = load_data(dataset_name,va_ratio=0.3)
#x_train,y_train,x_va,y_va,x_te,y_te = load_dataset_svhn(va_ratio=0.3)


print("x_train, nr sample {}, nr feature {}".format(x_train.shape[0],x_train.shape[1]))
print("x_va,    nr sample {}, nr feature {}".format(x_va.shape[0],x_va.shape[1]))
print("x_te,    nr sample {}, nr feature {}".format(x_te.shape[0],x_te.shape[1]))
print("Tr: Pos {} Neg {}".format(y_train[y_train==1].shape[0],y_train[y_train==0].shape[0]))
print("Va: Pos {} Neg {}".format(y_va[y_va==1].shape[0],y_va[y_va==0].shape[0]))
print("Te: Pos {} Neg {}".format(y_te[y_te==1].shape[0],y_te[y_te==0].shape[0]))
print("Load data, cost {:.1f} sec".format(time.time()-start_time))



# get the subset samples number
num_tr_sample = x_train.shape[0]
obj_sample_size = int(sample_ratio * num_tr_sample)
# flip labels
idxs = np.arange(y_train.shape[0])
np.random.shuffle(idxs)
num_flip = int(flip_ratio * len(idxs))
y_train[idxs[:num_flip]] = np.logical_xor(np.ones(num_flip), y_train[idxs[:num_flip]]).astype(int)




# define the full-set-model \hat{\theta}
clf = LogisticRegression(
        C = C,
        fit_intercept=False,
        tol = 1e-7,
        solver="liblinear",
        multi_class="ovr",
        max_iter=100,
        warm_start=False,
        verbose=0,
        )
clf.fit(x_train,y_train)
# on Va
y_va_pred = clf.predict_proba(x_va)[:,1]
full_logloss = log_loss(y_va,y_va_pred)
weight_ar = clf.coef_.flatten()
# on Te
y_te_pred = clf.predict_proba(x_te)[:,1]
full_te_logloss = log_loss(y_te,y_te_pred)
full_te_auc = roc_auc_score(y_te, y_te_pred)
y_te_pred = clf.predict(x_te)
full_te_acc = (y_te == y_te_pred).sum() / y_te.shape[0]


# print full-set-model results
print("[FullSet] Va logloss {:.6f}".format(full_logloss))
print("[FullSet] Te logloss {:.6f}".format(full_te_logloss))

# get time cost for computing the IF
if_start_time = time.time()
# building precoditioner
test_grad_loss_val = grad_logloss_theta_lr(y_va, y_va_pred, x_va, weight_ar, C, False, 0.1 / (num_tr_sample * C))
tr_pred = clf.predict_proba(x_train)[:, 1]
batch_size = 10000
M = None
total_batch = int(np.ceil(num_tr_sample / float(batch_size)))
for idx in range(total_batch):
        batch_tr_grad = batch_grad_logloss_lr(y_train[idx * batch_size:(idx + 1) * batch_size],
                                              tr_pred[idx * batch_size:(idx + 1) * batch_size],
                                              x_train[idx * batch_size:(idx + 1) * batch_size],
                                              weight_ar,
                                              C,
                                              False,
                                              1.0)

        sum_grad = batch_tr_grad.multiply(x_train[idx * batch_size:(idx + 1) * batch_size]).sum(0)
        if M is None:
                M = sum_grad
        else:
                M = M + sum_grad

M = M + 0.1 / (num_tr_sample * C) * np.ones(x_train.shape[1])
M = np.array(M).flatten()
# computing the inverse Hessian-vector-product
iv_hvp = inverse_hvp_lr_newtonCG(x_train, y_train, tr_pred, test_grad_loss_val, C, True, 1e-5, True, M,
                                 0.1 / (num_tr_sample * C))
# get influence score
total_batch = int(np.ceil(x_train.shape[0] / float(batch_size)))
predicted_loss_diff = []
for idx in range(total_batch):
        train_grad_loss_val = batch_grad_logloss_lr(y_train[idx * batch_size:(idx + 1) * batch_size],
                                                    tr_pred[idx * batch_size:(idx + 1) * batch_size],
                                                    x_train[idx * batch_size:(idx + 1) * batch_size],
                                                    weight_ar,
                                                    C,
                                                    False,
                                                    1.0)

        predicted_loss_diff.extend(np.array(train_grad_loss_val.dot(iv_hvp)).flatten())

predicted_loss_diffs = np.asarray(predicted_loss_diff)
duration = time.time() - if_start_time
print("The Influence function's computation completed, cost {:.1f} sec".format(duration))


print("=="*30)
print("IF Stats: mean {:.10f}, max {:.10f}, min {:.10f}".format(
    predicted_loss_diffs.mean(), predicted_loss_diffs.max(), predicted_loss_diffs.min())
)


# build sampling probability
phi_ar = -predicted_loss_diffs



tmp_list = sorted(phi_ar)

drop=0.0
correction = 1
dropsize = int(drop * num_tr_sample)
correction_size = int(correction*num_tr_sample)
drop_inf = tmp_list[:dropsize]
correction_inf1 = tmp_list[-correction_size:]

alpha = args.alpha
correction_inf = []
for i in correction_inf1:
     if i > alpha:
        correction_inf.append(i)

drop_index = []
for i in drop_inf:
        idx = np.argwhere(phi_ar==i)
        idx = idx.item()
        drop_index.append(idx)

correction_index=[]
for i in correction_inf:
        idx = np.argwhere(phi_ar==i)
        for id in idx:
            id = id.item()
            correction_index.append(id)
print("Relabeling index:",correction_index)
length = len(correction_index)
print("Totaling relabeling number:",length)

# RDIA
a = np.arange(y_train.shape[0])
new_a = a

sb_x_train = x_train[new_a]
sb_y_train1 = y_train.copy()
sb_y_train1[correction_index] = np.logical_xor(np.ones(len(correction_index)), sb_y_train1[correction_index]).astype(int)
sb_y_train = sb_y_train1


# Train the subset-model \tilde{\theta}
clf.fit(sb_x_train,sb_y_train)
y_va_pred = clf.predict_proba(x_va)[:,1]
sb_logloss = log_loss(y_va, y_va_pred)
sb_weight = clf.coef_.flatten()
diff_w_norm = np.linalg.norm(weight_ar - sb_weight)
sb_size = sb_x_train.shape[0]
y_te_pred = clf.predict_proba(x_te)[:,1]
sb_te_logloss = log_loss(y_te,y_te_pred)
sb_te_auc = roc_auc_score(y_te, y_te_pred)
y_te_pred = clf.predict(x_te)
sb_te_acc = (y_te == y_te_pred).sum() / y_te.shape[0]



# baseline: random sampling
u_idxs = np.arange(x_train.shape[0])
uniform_idxs = np.random.choice(u_idxs,obj_sample_size,replace=False)
us_x_train = x_train[uniform_idxs]
us_y_train = y_train[uniform_idxs]
clf.fit(us_x_train, us_y_train)
y_va_pred = clf.predict_proba(x_va)[:,1]
us_logloss = log_loss(y_va, y_va_pred)
us_size = us_x_train.shape[0]
y_te_pred = clf.predict_proba(x_te)[:,1]
us_te_logloss = log_loss(y_te,y_te_pred)
us_te_auc = roc_auc_score(y_te, y_te_pred)
y_te_pred = clf.predict(x_te)
us_te_acc = (y_te == y_te_pred).sum() / y_te.shape[0]



print("=="*30)
print("Result Summary on Va")
print("[RDIA]  logloss {:.6f}, # {}".format(sb_logloss,sb_size))
print("[Random]   logloss {:.6f}, # {}".format(us_logloss,us_size))
print("[Full]     logloss {:.6f}, # {}".format(full_logloss,num_tr_sample))
print("Result Summary on Te")
print("[RDIA]  logloss {:.6f}, # {}".format(sb_te_logloss,sb_size))
print("[Random]   logloss {:.6f}, # {}".format(us_te_logloss,us_size))
print("[Full]     logloss {:.6f}, # {}".format(full_te_logloss,num_tr_sample))
print("=="*30)

print("=="*30)
print("Result Summary on Te (ACC and AUC)")
print("[RDIA]  acc {:.6f}, auc {:.6f} # {}".format(sb_te_acc,sb_te_auc, sb_size))
print("[Random]   acc {:.6f}, auc {:.6f} # {}".format(us_te_acc,us_te_auc, us_size))
print("[Full]     acc {:.6f}, auc {:.6f} # {}".format(full_te_acc,full_te_auc, num_tr_sample))
print("=="*30)

