# -*- coding:utf-8 -*-
from __future__ import print_function 
import os
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
from data.cifar import CIFAR10, CIFAR100
from data.mnist import MNIST
from data.newsgroups import NewsGroups
from data.torchlist import ImageFilelist
from model import MLPNet, CNN_small, CNN, NewsNet
from preact_resnet import PreActResNet18
import argparse, sys
import numpy as np
import datetime
import shutil

from loss import loss_RDIA,loss_warmup

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type = float, default = 0.001)
parser.add_argument('--result_dir', type = str, help = 'dir to save result txt files', default = '/home/kongshuming/RDIA-LS/results')
parser.add_argument('--noise_rate', type = float, help = 'corruption rate, should be less than 1', default = 0.8)
parser.add_argument('--forget_rate', type = float, help = 'forget rate', default = None)
parser.add_argument('--noise_type', type = str, help='[pairflip, symmetric]', default='symmetric')
parser.add_argument('--num_gradual', type = int, default = 10, help='how many epochs for linear drop rate. This parameter is equal to Ek for lambda(E) in the paper.')
parser.add_argument('--dataset', type = str, help = 'mnist, cifar10, cifar100, or imagenet_tiny', default = 'mnist')
parser.add_argument('--n_epoch', type=int, default=200)
parser.add_argument('--optimizer', type = str, default='adam')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--gamma', type=float, default=0.5)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--print_freq', type=int, default=50)
parser.add_argument('--num_workers', type=int, default=4, help='how many subprocesses to use for data loading')
parser.add_argument('--epoch_decay_start', type=int, default=80)
parser.add_argument('--model_type', type = str, help='[coteaching, coteaching_plus]', default='coteaching_plus')
parser.add_argument('--fr_type', type = str, help='forget rate type', default='type_1')
parser.add_argument('--gpu', type=int, default=0)
args = parser.parse_args()




# Seed
torch.manual_seed(args.seed)
if args.gpu is not None:
    device = torch.device('cuda:{}'.format(args.gpu))
    torch.cuda.manual_seed(args.seed)

else:
    device = torch.device('cpu')
    torch.manual_seed(args.seed)

# Hyper Parameters
batch_size = 128
learning_rate = args.lr
noise_rate = args.noise_rate
gamma = args.gamma
weight = args.weight_decay


# load dataset
if args.dataset=='mnist':
    input_channel = 1
    init_epoch = 10
    num_classes = 10
    args.n_epoch = 200
    train_dataset = MNIST(root='./data/',
                                download=True,
                                train=True,
                                transform=transforms.ToTensor(),
                                noise_type=args.noise_type,
                                noise_rate=args.noise_rate
                                )

    test_dataset = MNIST(root='./data/',
                               download=True,
                               train=False,
                               transform=transforms.ToTensor(),
                               noise_type=args.noise_type,
                               noise_rate=args.noise_rate
                                )

if args.dataset=='cifar10':
    input_channel=3
    init_epoch = 10
    num_classes = 10
    args.n_epoch = 200
    train_dataset = CIFAR10(root='./data/',
                                download=True,
                                train=True,
                                transform=transforms.ToTensor(),
                                noise_type=args.noise_type,
                                noise_rate=args.noise_rate
                                )

    test_dataset = CIFAR10(root='./data/',
                                download=True,
                                train=False,
                                transform=transforms.ToTensor(),
                                noise_type=args.noise_type,
                                noise_rate=args.noise_rate
                                )

if args.dataset=='cifar100':
    input_channel=3
    init_epoch = 20
    num_classes = 100
    args.n_epoch = 200
    train_dataset = CIFAR100(root='./data/',
                                download=True,
                                train=True,
                                transform=transforms.ToTensor(),
                                noise_type=args.noise_type,
                                noise_rate=args.noise_rate
                                )

    test_dataset = CIFAR100(root='./data/',
                                download=True,
                                train=False,
                                transform=transforms.ToTensor(),
                                noise_type=args.noise_type,
                                noise_rate=args.noise_rate
                                )


if args.dataset=='news':
    init_epoch=0
    train_dataset = NewsGroups(root='./data/',
                                train=True,
                                transform=transforms.ToTensor(),
                                noise_type=args.noise_type,
                                noise_rate=args.noise_rate
                                )

    test_dataset = NewsGroups(root='./data/',
                               train=False,
                               transform=transforms.ToTensor(),
                               noise_type=args.noise_type,
                               noise_rate=args.noise_rate
                                )
    num_classes=train_dataset.num_classes

if args.dataset == 'imagenet_tiny':
    init_epoch = 100
    #data_root = '/home/xingyu/Data/phd/data/imagenet-tiny/tiny-imagenet-200'
    data_root = 'data/imagenet-tiny/tiny-imagenet-200'
    train_kv = "train_noisy_%s_%s_kv_list.txt" % (args.noise_type, args.noise_rate)
    test_kv = "val_kv_list.txt"

    normalize = transforms.Normalize(mean=[0.4802, 0.4481, 0.3975],
                                     std =[0.2302, 0.2265, 0.2262])

    train_dataset = ImageFilelist(root=data_root, flist=os.path.join(data_root, train_kv),
               transform=transforms.Compose([transforms.RandomResizedCrop(56),
               transforms.RandomHorizontalFlip(),
               transforms.ToTensor(),
               normalize,
       ]))

    test_dataset = ImageFilelist(root=data_root, flist=os.path.join(data_root, test_kv),
               transform=transforms.Compose([transforms.Resize(64),
               transforms.CenterCrop(56),
               transforms.ToTensor(),
               normalize,
       ]))

if args.forget_rate is None:
    forget_rate=args.noise_rate
else:
    forget_rate=args.forget_rate

if args.dataset == 'imagenet_tiny':
    noise_or_not = np.load(os.path.join(data_root, 'noise_or_not_%s_%s.npy' %(args.noise_type, args.noise_rate)))
else:
    noise_or_not = train_dataset.noise_or_not

# Adjust learning rate and betas for Adam Optimizer
mom1 = 0.9
mom2 = 0.1
alpha_plan = [learning_rate] * args.n_epoch
beta1_plan = [mom1] * args.n_epoch
for i in range(args.epoch_decay_start, args.n_epoch):
    alpha_plan[i] = float(args.n_epoch - i) / (args.n_epoch - args.epoch_decay_start) * learning_rate
    beta1_plan[i] = mom2

def adjust_learning_rate(optimizer, epoch):
    for param_group in optimizer.param_groups:
        param_group['lr']=alpha_plan[epoch]
        param_group['betas']=(beta1_plan[epoch], 0.999)

# define drop rate schedule
def gen_forget_rate(fr_type='type_1'):
    if fr_type=='type_1':
        rate_schedule = np.ones(args.n_epoch)*forget_rate
        rate_schedule[:args.num_gradual] = np.linspace(0, forget_rate, args.num_gradual)

    #if fr_type=='type_2':
    #    rate_schedule = np.ones(args.n_epoch)*forget_rate
    #    rate_schedule[:args.num_gradual] = np.linspace(0, forget_rate, args.num_gradual)
    #    rate_schedule[args.num_gradual:] = np.linspace(forget_rate, 2*forget_rate, args.n_epoch-args.num_gradual)

    return rate_schedule

rate_schedule = gen_forget_rate(args.fr_type)



save_dir = args.result_dir +'/' +args.dataset

if not os.path.exists(save_dir):
    os.system('mkdir -p %s' % save_dir)

model_str = args.dataset + '_%s_' % args.model_type + args.noise_type + '_' + str(args.noise_rate)

txtfile = save_dir + "/" + model_str + "RDIA_0.2.txt"
nowTime=datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')



def accuracy(logit, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    ok=0
    output = F.softmax(logit, dim=1)
    maxk = max(topk)
    batch_size = target.size(0)

    _, predicted = torch.max(output.data, 1)
    # add the accuracy
    ok += (predicted == target).sum()
    return ok

# Train the Model
def train(train_loader,epoch, model1, optimizer1, model2, optimizer2,noise_rate,gamma):
    print('Training %s...' % model_str)

    train_total=0
    train_correct=0
    train_total2=0
    train_correct2=0
    total_disagreement = 0
    pure_ratio_1_list = []
    pure_ratio_2_list = []
    for i, (data, labels, indexes) in enumerate(train_loader):
        ind=indexes.cpu().numpy().transpose()

        labels = Variable(labels.long()).to(device)

        if args.dataset=='news':
            data = Variable(data.long()).to(device)
        else:
            data = Variable(data).to(device)
        # Forward + Backward + Optimize
        logits1=model1(data)
        prec1  = accuracy(logits1, labels, topk=(1, ))
        train_total+=1
        train_correct+=prec1

        logits2 = model2(data)
        prec2  = accuracy(logits2, labels, topk=(1, ))
        train_total2+=1
        train_correct2+=prec2
        if epoch < init_epoch:
            loss_1,pure_ratio_1= loss_warmup(logits1, labels, rate_schedule[epoch], ind,noise_or_not)
            pure_ratio_1=0
        else:
            if args.model_type=='coteaching_plus':
                loss_1, pure_ratio_1 = loss_RDIA(logits1, labels, rate_schedule[epoch], ind, noise_or_not,epoch,noise_rate,gamma)

        optimizer1.zero_grad()
        loss_1.backward()
        optimizer1.step()
        pure_ratio_1_list.append(100 * pure_ratio_1)
        if (i+1) % args.print_freq == 0:
            print('Epoch [%d/%d], Iter [%d/%d] Training Accuracy1: %.4F, Training Accuracy2: %.4f'
                  %(epoch+1, args.n_epoch, i+1, len(train_dataset)//batch_size, prec1, prec2))
    disagreement = float(total_disagreement)/float(train_total)
    train_acc1=float(train_correct)/float(train_total)
    train_acc2=float(train_correct2)/float(train_total2)
    return train_acc1, train_acc2,pure_ratio_1_list, pure_ratio_2_list,disagreement

# Evaluate the Model
def evaluate(test_loader, model1, model2):
    print('Evaluating %s...' % model_str)
    model1.eval()    # Change model to 'eval' mode.
    correct1 = 0
    total1 = 0
    for data, labels, _ in test_loader:
        if args.dataset=='news':
            data = Variable(data.long()).to(device)
        else:
            data = Variable(data).to(device)
        logits1 = model1(data)
        outputs1 = F.softmax(logits1, dim=1)
        _, pred1 = torch.max(outputs1.data, 1)
        total1 += labels.size(0)
        correct1 += (pred1.cpu() == labels.long()).sum()

    model2.eval()    # Change model to 'eval' mode
    correct2 = 0
    total2 = 0
    for data, labels, _ in test_loader:
        if args.dataset=='news':
            data = Variable(data.long()).to(device)
        else:
            data = Variable(data).to(device)
        logits2 = model2(data)
        outputs2 = F.softmax(logits2, dim=1)
        _, pred2 = torch.max(outputs2.data, 1)
        total2 += labels.size(0)
        correct2 += (pred2.cpu() == labels.long()).sum()

    acc1 = 100*float(correct1)/float(total1)
    acc2 = 100*float(correct2)/float(total2)
    return acc1, acc2

def main():
    # Data Loader (Input Pipeline)
    print('loading dataset...')
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               num_workers=args.num_workers,
                                               drop_last=True,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              num_workers=args.num_workers,
                                              drop_last=True,
                                              shuffle=False)
    # Define models
    print('building model...')
    if args.dataset == 'mnist':
        clf1 = MLPNet()
    if args.dataset == 'cifar10':
        clf1 = CNN(n_outputs=num_classes)
    if args.dataset == 'cifar100':
        clf1 = CNN(n_outputs=num_classes)
    if args.dataset=='news':
        clf1 = NewsNet(weights_matrix=train_dataset.weights_matrix, num_classes=num_classes)
    if args.dataset=='imagenet_tiny':
        clf1 = PreActResNet18(num_classes=200)

    clf1.to(device)
    print(clf1.parameters)
    optimizer1 = torch.optim.Adam(clf1.parameters(), lr=learning_rate,weight_decay=weight)

    if args.dataset == 'mnist':
        clf2 = MLPNet()
    if args.dataset == 'cifar10':
        clf2 = CNN(n_outputs=num_classes)
    if args.dataset == 'cifar100':
        clf2 = CNN(n_outputs=num_classes)
    if args.dataset=='news':
        clf2 = NewsNet(weights_matrix=train_dataset.weights_matrix, num_classes=num_classes)
    if args.dataset=='imagenet_tiny':
        clf2 = PreActResNet18(num_classes=200)

    clf2.to(device)
    print(clf2.parameters)
    optimizer2 = torch.optim.Adam(clf2.parameters(), lr=learning_rate,weight_decay=weight)
    optimizer = torch.optim.Adam(list(clf1.parameters()) + list(clf2.parameters()),
                                 lr=learning_rate)
    myfile = open(txtfile,'w')
    myfile.write('epoch train_acc1 train_acc2 test_acc1 test_acc2\n')

    epoch=0
    train_acc1=0
    train_acc2=0

    # evaluate models with random weights
    test_acc1, test_acc2=evaluate(test_loader, clf1, clf2)
    print('Epoch [%d/%d] Test Accuracy on the %s test data: Model1 %.4f %% %%' % (epoch+1, args.n_epoch, len(test_dataset), test_acc1))
    # save results
    with open(txtfile, "a") as myfile:
        myfile.write(str(int(epoch)) + ' '  + str(train_acc1) +' '  + str(train_acc2) +' '  + str(test_acc1) + " " + str(test_acc2)  + "\n")

    # training
    for epoch in range(1, args.n_epoch):
        # train models
        clf1.train()
        clf2.train()

        adjust_learning_rate(optimizer1, epoch)
        adjust_learning_rate(optimizer2, epoch)
        a = epoch/100

        train_acc1, train_acc2,pure_ratio_1_list, pure_ratio_2_list,disagreement = train(train_loader, epoch, clf1, optimizer1, clf2, optimizer2,noise_rate,gamma)

        # evaluate models
        test_acc1, test_acc2 = evaluate(test_loader, clf1, clf2)
        if pure_ratio_1_list is None or len(pure_ratio_1_list) == 0:
            print(
                'Epoch [%d/%d] Test Accuracy on the %s test images: Model1 %.4f %% ' % (
                    epoch + 1, args.n_epoch, len(test_dataset), test_acc1))
        else:
            # save results
            mean_pure_ratio1 = sum(pure_ratio_1_list) / len(pure_ratio_1_list)
            print(
                'Epoch [%d/%d] Test Accuracy on the %s test images: Model1 %.4f  %%, Pure Ratio 1 %.4f %% %%' % (
                    epoch + 1, args.n_epoch, len(test_dataset), test_acc1, mean_pure_ratio1))
            with open(txtfile, "a") as myfile:
                myfile.write(
                    str(int(epoch)) + ' ' + str(train_acc1) + ' ' + str(train_acc2) + ' ' + str(test_acc1) + " " + str(
                        test_acc2) + " " + str(mean_pure_ratio1) + " " + str(disagreement)+ " "+"\n")

if __name__=='__main__':
    main()
