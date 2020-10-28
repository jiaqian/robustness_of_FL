
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torch.autograd import Variable
from sklearn.metrics import accuracy_score
import torch.optim as optim
import torchvision
import itertools
import os
import gzip
from sklearn.utils import shuffle
from network import Model


get_slice = lambda i, size: range(i * size, (i + 1) * size)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def random_sample(m,X,y):
  index = np.random.choice(X.shape[0], m, replace=False)
  return X[index],y[index]

def ini_model(opt):
    nb_classes =10
    if opt.dataset=='mfashion':
        X_train_All,y_train_All,X_test,y_test = fmnist_loader()
    else:
        raise Exception('should be mfashion')
    X,y,X_tr,y_tr = X_train_All[:10000,:],y_train_All[:10000],X_train_All[10000:,:],y_train_All[10000:]
    m=opt.ini_number
    X_ini,y_ini = random_sample(m,X,y)
    X_ini,y_ini = shuffle(X_ini,y_ini)
    print("initial size ",X_ini.shape)
    return X_ini,y_ini,X_test,y_test,X_train_All,y_train_All

def ini_model_train(opt):
    X_ini,y_ini,X_test,y_test,X_train_All,y_train_All=ini_model(opt)
    mod = Model().to(device)
    optimizer= optim.SGD(mod.parameters(), lr=opt.ini_lr)
    criterion = nn.CrossEntropyLoss()
    num_batches_train = X_ini.shape[0]//opt.ini_batch_size
    mod.train()
    for i in range(opt.ini_epoch):
        loss=0
        for j in range(num_batches_train):
            slce = get_slice(j,opt.ini_batch_size)
            X_tra = torch.from_numpy(X_ini[slce]).float().to(device)
            Y_tra = torch.from_numpy(y_ini[slce]).long().to(device)
            optimizer.zero_grad()
            out = mod(X_tra)
            batch_loss = criterion(out,Y_tra)
            batch_loss.backward()
            optimizer.step()
            loss+=batch_loss
        mod.eval()
        acc=test_without_dropout(X_test,y_test,mod,device)
        print('\n[{}/{} epoch], training loss:{:.4f}, test accuracy is:{} \n'.format(i,opt.ini_epoch,loss.item()/num_batches_train,acc))
        if i+1 == opt.ini_epoch:
            for d in range(opt.num_dev):
                torch.save({'epoch':i,
                    'model_state_dict':mod.state_dict(),
                    'optimizer_state_dict':optimizer.state_dict(),
                    'loss':loss.item()},os.path.join(opt.ini_model_path,'device'+str(d),"ini.model.pth.tar"))
            torch.save({'epoch':i,'model_state_dict':mod.state_dict(),'optimizer_state_dict':optimizer.state_dict(), 'loss':loss.item()},opt.ini_model_path)
    return X_test,y_test,X_train_All,y_train_All




def sample_generator(L,nr_dev,X,y,target=10,nr_dist=None):
    conx = []
    cony = []
    print('num of devices is: ', nr_dev)
    #nr_dev= len(nr_dist)
    if nr_dist is None:
        nr_dist = np.random.randint(1, target, size=(nr_dev,))
        while sum(nr_dist) != target: 
            nr_dist = np.random.randint(1, target, size=(4,))
    print(nr_dist)
    record = []
    for i in range(nr_dev):
        ls = L[:nr_dist[i]]
        L = L[nr_dist[i]:]
        #print('dev {} has {} labels'.format(i+1,ls))
        for j,l in enumerate(ls):
            #print('dev {} has {} labels, {}th label is {} '.format(i+1,ls,j+1,l))
            #print('label {} has {} data'.format(l,X[y==l,:,:,:].shape[0]))
            conx.append(X[y==l,:,:,:])
            cony.append(y[y==l])
            record.append(X[y==l,:,:,:].shape[0])
    return conx,cony,record,nr_dist

def unbalanced_data(X_train_All,y_train_All,opt,nr_dist):
    if len(nr_dist) != 4:
        raise ValueError('device number should be four')
    L = np.unique(y_train_All)
    X_tr,y_tr = X_train_All[10000:,:],y_train_All[10000:]
    conx,cony,record,tmp = sample_generator(L,opt.num_dev,X_tr,y_tr,nr_dist=nr_dist)
    conx = np.vstack(conx)
    cony = np.hstack(cony)
    tmp = list(np.cumsum(np.array(tmp)))
    X1,y1 = conx[:sum(record[:tmp[0]]),:,:,:],cony[:sum(record[:tmp[0]])]
    X2,y2 = conx[sum(record[:tmp[0]]):sum(record[:tmp[1]]),:,:,:],cony[sum(record[:tmp[0]]):sum(record[:tmp[1]])]
    X3,y3 = conx[sum(record[:tmp[1]]):sum(record[:tmp[2]]),:,:,:],cony[sum(record[:tmp[1]]):sum(record[:tmp[2]])]
    X4,y4 = conx[sum(record[:tmp[2]]):sum(record[:tmp[3]]),:,:,:],cony[sum(record[:tmp[2]]):sum(record[:tmp[3]])]
    X1,y1=shuffle(X1,y1)
    X2,y2=shuffle(X2,y2)
    X3,y3=shuffle(X3,y3)
    X4,y4=shuffle(X4,y4)
    print('device1 has {} data'.format(X1.shape[0]))
    print('device2 has {} data'.format(X2.shape[0]))
    print('device3 has {} data'.format(X3.shape[0]))
    print('device4 has {} data'.format(X4.shape[0]))
    return X1,y1,X2,y2,X3,y3,X4,y4

    
def en_ave(mod1,mod2,mod3,mod4,X_test,y_test,state):
    print('=> load average ensemble')
    mod = Model().to(device)
    for p, p1, p2, p3, p4 in zip(mod.parameters(), mod1.parameters(), mod2.parameters(),mod3.parameters(), mod4.parameters()):
        p.data.copy_(p1.data.mul(0.25).add(p2.data.mul(0.25)).add(p3.data.mul(0.25)).add(p4.data.mul(0.25)))
    acc = test_without_dropout(X_test,y_test,mod,device)
    path = os.path.join('exp','ensemble.')+str(state['itr'])+'epoch.'+str(state['acq'])+'acq.pth.tar'
    state['rep'] = path
    torch.save({'model_state_dict':mod.state_dict()}, state['rep'])
    return mod,acc

def en_opt(mod1,mod2,mod3,mod4,acc1,acc2,acc3,acc4):
    print('=> load optimal ensemble')
    M = []
    M.append(mod1)
    M.append(mod2)
    M.append(mod3)
    M.append(mod4)
    ind_max = np.argmax([acc1,acc2,acc3,acc4])
    acc = np.max([acc1,acc2,acc3,acc4])
    mod = M[ind_max]
    return mod,acc

def en_mix(mod1,mod2,mod3,mod4,X_test,y_test,acc1,acc2,acc3,acc4,state):
    print('=> load mix ensemble')
    mod_ave, acc_ave = en_ave(mod1,mod2,mod3,mod4,X_test,y_test)
    mod_opt, acc_opt = en_opt(mod1,mod2,mod3,mod4,acc1,acc2,acc3,acc4)
    print('ensembled acc_ave: {:.3f}, ensembled acc_opt: {:.3f}'.format(acc_ave,acc_opt))
    if acc_ave >= acc_opt:
        path = os.path.join('exp','ensemble.')+str(state['itr'])+'epoch.'+str(state['acq'])+'acq.pth.tar'
        state['rep'] = path
        torch.save({'model_state_dict':mod_ave.state_dict()}, state['rep'])
        return mod_ave,acc_ave
    else:
        path = os.path.join('exp','ensemble.')+str(state['itr'])+'epoch.'+str(state['acq'])+'acq.pth.tar'
        state['rep'] = path
        torch.save({'model_state_dict':mod_opt.state_dict()}, state['rep'])
        return mod_opt,acc_opt


def test_without_dropout(X_test,y_test,mod,device):

    mod.eval()
    with torch.no_grad():
        X_va = torch.from_numpy(X_test).float().to(device)
        Y_va = torch.from_numpy(y_test).long().to(device)
        output= mod(X_va)
        test_loss = F.cross_entropy(output,Y_va).item()/X_test.shape[0]
        preds = torch.max(output,1)[1]
        acc = accuracy_score(Y_va.cpu(),preds.cpu())
    return acc

def ensemble(state,X_test,y_test,g):
    mod1 = Model().to(state['device'])
    mod2 = Model().to(state['device'])
    mod3 = Model().to(state['device'])
    mod4 = Model().to(state['device'])
    mod = Model().to(state['device'])
    mod1.load_state_dict(torch.load(state['path1'])['model_state_dict'])
    mod2.load_state_dict(torch.load(state['path2'])['model_state_dict'])
    mod3.load_state_dict(torch.load(state['path3'])['model_state_dict'])
    mod4.load_state_dict(torch.load(state['path4'])['model_state_dict'])

    for p, p1, p2, p3, p4 in zip(mod.parameters(), mod1.parameters(), mod2.parameters(), mod3.parameters(), mod4.parameters()):
        p.data.copy_(p1.data.mul(0.25).add(p2.data.mul(0.25)).add(p3.data.mul(0.25)).add(p4.data.mul(0.25)))
    mod.state_dict()
    acc = test_with_dropout(X_test,y_test,mod,state['device'],state['cuda'])
    path = g+str(state['itr'])+'epoch.'+str(state['acq'])+'acq.pth.tar'
    state['rep'] = path
    torch.save({'model_state_dict':mod.state_dict()}, state['rep'])

    return mod,acc

def random_sample(m,X,y):
  index = np.random.choice(X.shape[0], m, replace=False)
  return X[index],y[index]

def ini_train(X_ini,y_ini,X_te,y_te,epochs,paths,device,batch_size,lr,momentum,arr_drop):
    mod = Model(arr_drop).to(device)
    optimizer= optim.SGD(mod.parameters(), lr=lr,momentum=momentum )
    criterion = nn.CrossEntropyLoss()
    #batch_size = 200
    num_batches_train = X_ini.shape[0]//batch_size
    print("number of batch ",num_batches_train)
    mod.train()
    for i in range(epochs):
        loss=0
        for j in range(num_batches_train):
            slce = get_slice(j,batch_size)
            X_tra = torch.from_numpy(X_ini[slce]).float().to(device)
            Y_tra = torch.from_numpy(y_ini[slce]).long().to(device)
            optimizer.zero_grad()
            out = mod(X_tra)
            batch_loss = criterion(out,Y_tra)
            batch_loss.backward()
            optimizer.step()
            loss+=batch_loss
        mod.eval()
        with torch.no_grad():
            X_va = torch.from_numpy(X_te).float().to(device)
            Y_va = torch.from_numpy(y_te).long().to(device)
            output= mod(X_va)
            preds = torch.max(output,1)[1]
            acc = accuracy_score(Y_va,preds)
        print('\n[{}/{} epoch], training loss:{:.4f}, test accuracy is:{} \n'.format(i,epochs,loss.item()/num_batches_train,acc))
    if i+1 == epochs:
        for path in paths:
            torch.save({
                'epoch': i,
                'model_state_dict': mod.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item()}, os.path.join(path,"ini.model.pth.tar"))      
    return mod


def random_run(acquisition_iterations,X_Pool,y_Pool,pool_subset,dropout_iterations,
                nb_classes,Queries,X_test,y_test,rep,X_old,y_old,device,itr,cuda,g):
        mod = Model().to(device)
        if cuda:
            cp = torch.load(rep)
            print("\n ********load gpu version******* \n")
        else:
            cp = torch.load(rep, map_location='cpu')
        mod.load_state_dict(cp['model_state_dict'])
        optimizer = optim.Adam(mod.parameters(), lr=0.001,weight_decay=0.5)#,weight_decay=0.5
        #optimizer = optim.SGD(mod.parameters(), lr=0.001,weight_decay=0.5)
        optimizer.load_state_dict(cp['optimizer_state_dict'])
        criterion = nn.CrossEntropyLoss()
        X_train = np.empty([0,1,28,28])
        y_train = np.empty([0,])
        AA = []
        losses_train = []
        #acc = test(test_loader,mod,device,cuda)
        acc = test(X_test,y_test,mod,device,cuda)
        AA.append(acc)  
        print('initial test accuracy: ',acc)
        for i in range(acquisition_iterations):
            pool_subset_dropout = np.asarray(random.sample(range(0,X_Pool.shape[0]), pool_subset))
            X_Pool_Dropout = X_Pool[pool_subset_dropout, :, :, :]
            y_Pool_Dropout = y_Pool[pool_subset_dropout]
            
            x_pool_index = np.random.choice(X_Pool_Dropout.shape[0], Queries, replace=False)
            Pooled_X = X_Pool_Dropout[x_pool_index, :,:,:]
            Pooled_Y = y_Pool_Dropout[x_pool_index]

            delete_Pool_X = np.delete(X_Pool, (pool_subset_dropout), axis=0)
            delete_Pool_Y = np.delete(y_Pool, (pool_subset_dropout), axis=0)

            delete_Pool_X_Dropout = np.delete(X_Pool_Dropout, (x_pool_index), axis=0)
            delete_Pool_Y_Dropout = np.delete(y_Pool_Dropout, (x_pool_index), axis=0)

            X_Pool = np.concatenate((delete_Pool_X, delete_Pool_X_Dropout), axis=0)
            y_Pool = np.concatenate((delete_Pool_Y, delete_Pool_Y_Dropout), axis=0)
            print('updated pool size is ',X_Pool.shape[0])


            X_train = np.concatenate((X_train, Pooled_X), axis=0)
            y_train = np.concatenate((y_train, Pooled_Y), axis=0)
            print('number of data points from pool',X_train.shape[0])

            batch_size = 100
            X = np.vstack((X_old,Pooled_X))
            y = np.hstack((y_old,Pooled_Y))
            X,y = shuffle(X,y)
            num_batch = X.shape[0]//batch_size
            print("number of batch: ",num_batch)
            mod.train()
            for h in range(itr):
                losses = 0
                for j in range(num_batch):
                    slce = get_slice(j,batch_size)
                    X_fog_ = torch.from_numpy(X[slce]).float().to(device)
                    y_fog_ = torch.from_numpy(y[slce]).long().to(device)
                    optimizer.zero_grad()
                    out = mod(X_fog_)
                    train_loss = criterion(out,y_fog_)
                    losses += train_loss
                    train_loss.backward()
                    optimizer.step()
                losses_train.append(losses.item()/num_batch)
            acc = test(X_test,y_test,mod,device,cuda)
            print('test accuracy: ',acc)
            AA.append(acc)
        torch.save({'model_state_dict':mod.state_dict(),'optimizer_state_dict':optimizer.state_dict()}, g)
        return AA,mod,X_train,y_train,losses_train,optimizer