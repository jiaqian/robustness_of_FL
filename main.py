import argparse
import os
from network import *
from unbalanced_utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='mfashion')
# parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--cuda', type=bool, default=True)
parser.add_argument('--ini_model', type=bool, default=True)
parser.add_argument('--ini_number', type=int, default=400)
parser.add_argument('--ini_epoch', type=int, default=50)
parser.add_argument('--ini_batch_size', type=int, default=200)
parser.add_argument('--ini_model_path', default='exp',help='create the corresponding folder')
parser.add_argument('--ini_lr', type=float, default=0.05)
parser.add_argument('--num_dev', type=int, default=4)
parser.add_argument('--batch_size', type=int, default=200)
parser.add_argument('--aq', type=int, default=10)
parser.add_argument('--Queries', type=int, default=400)

# parser.add_argument('--num_epochs', type=int, default=120)
get_slice = lambda i,size: range(i*size, (i+1)*size)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
opt = parser.parse_args()
nr_dist = [2,2,3,3]
if opt.ini_model:
	print('=> initialize model')
	X_test,y_test,X_train_All,y_train_All=ini_model_train(opt)
	print('complete the initial setting, and saved in {}'.format(opt.ini_model_path))
else:
	print('yet prepared ini model')
	X_ini,y_ini,X_test,y_test,X_train_All,y_train_All=ini_model(opt)
# divide data into four parts for distributed devices
X1,y1,X2,y2,X3,y3,X4,y4=unbalanced_data(X_train_All,y_train_All,opt,nr_dist)
itrs = [1,5,10,20,30,40,45]
acqs = [1,2,5,10]
state={}
state['aq']=10
state['Queries'] = 400
state['batch_size'] = 200
recorder = np.zeros((len(itrs)*len(acqs),5))
count=0

def train_with_pool(acq,X_pool,y_pool,state,pp,n,dd):
	mod = Model().to(device)
	mod.load_state_dict(torch.load(state['rep'])['model_state_dict'])#,map_location='cpu'
	optimizer = optim.SGD(mod.parameters(), lr = 0.05, momentum=0.95)
	optimizer.load_state_dict(torch.load(state['path1'])['optimizer_state_dict'])#map_location='cpu'
	criterion =  nn.CrossEntropyLoss()
	X_train = np.empty([0,1,28,28])
	y_train = np.empty([0,])
	A =[]
	loss_log = []
	acc = test_without_dropout(X_test,y_test,mod,device)
	print('initial test accuracy: ',acc)
	A.append(acc)
	for i in range(acq):
		index = np.random.choice(X_pool.shape[0], state['Queries'], replace=False)
		X = X_pool[index, :,:,:]
		y = y_pool[index]
		#print('pool size: ',X_pool.shape[0])
		X_pool = np.delete(X_pool, index, axis=0)
		y_pool = np.delete(y_pool, index, axis=0)
		print('updated pool size is {}'.format(X_pool.shape[0]))

		X_train = np.concatenate((X_train, X), axis=0)
		y_train = np.concatenate((y_train, y), axis=0)

		num_batch = X.shape[0]//state['batch_size']

		mod.train()
		for epoch in range(state['itr']):
			loss = 0
			for j in range(num_batch):
				slce = get_slice(j,state['batch_size'])
				X_batch = torch.from_numpy(X[slce]).float().to(device)
				y_batch = torch.from_numpy(y[slce]).long().to(device)
				optimizer.zero_grad()
				out = mod(X_batch)
				# print('out',out.size())
				# print(y_batch.size())
				bloss = criterion(out,y_batch)
				loss+=bloss.item()
				bloss.backward()
				optimizer.step()
			loss_log.append(loss/num_batch)
	# save model
		acc= test_without_dropout(X_test,y_test,mod,device)
		print('device{} in [{}/{}] acq, acc is {:.3f} '.format(n,i+1,acq,acc))

	state[pp] = os.path.join(opt.ini_model_path,dd,str(state['itr'])+'epoch.'+str(state['acq'])+'acq.pth.tar')
	print('update {} to {}'.format(pp,state[pp]))
	torch.save({
		'epoch':epoch,
		'model_state_dict':mod.state_dict(),
		'optimizer_state_dict':optimizer.state_dict(),
		'loss':loss_log[0]},state[pp])
	return mod, acc, X_pool, y_pool

for itr in itrs:
	state['itr'] = itr
	for acq in acqs:
		state['rep']  = os.path.join(opt.ini_model_path,'ini.model.pth.tar')
		state['acq'] = acq
		AA1,AA2,AA3,AA4,AA = [],[],[],[],[]
		X1_,y1_,X2_,y2_,X3_,y3_,X4_,y4_=X1,y1,X2,y2,X3,y3,X4,y4
		print('acq is {}, epoch is {} '.format(acq,itr))
		for r in range(state['aq']//acq):
			if r==0:
				if os.path.exists(os.path.join('exp','device1','ini.model.pth.tar')):
					state['path1'] = os.path.join('exp','device1','ini.model.pth.tar')
				elif os.path.exists(os.path.join('exp','device2','ini.model.pth.tar')):
					state['path2'] = os.path.join('exp','device2','ini.model.pth.tar')
				elif os.path.exists(os.path.join('exp','device3','ini.model.pth.tar')):
					state['path3'] = os.path.join('exp','device3','ini.model.pth.tar')
				elif os.path.exists(os.path.join('exp','device4','ini.model.pth.tar')):
					state['path4'] = os.path.join('exp','device4','ini.model.pth.tar')
			print('********** communication {}/{} rounds ***********\n'.format(r+1,state['aq']//acq))
			mod1, acc1,X1_,y1_  = train_with_pool(acq,X1_,y1_,state,'path1','1','device1')
			mod2, acc2,X2_,y2_  = train_with_pool(acq,X2_,y2_,state,'path2','2','device2')
			mod3, acc3,X3_,y3_ = train_with_pool(acq,X3_,y3_,state,'path3','3','device3')
			mod4, acc4,X4_,y4_ = train_with_pool(acq,X4_,y4_,state,'path4','4','device4')

			mod, acc = en_ave(mod1,mod2,mod3,mod4,X_test,y_test,state)
			AA1.append(acc1)
			AA2.append(acc2)
			AA3.append(acc3)
			AA4.append(acc4)
			AA.append(acc)
		recorder[count,:]=[acc1,acc2,acc3,acc4,acc]
		count+=1
		print('dev1 acc is {:.3f}, device2 acc is {:.3f},dev3 acc is {:.3f}, dev4 acc is {:.3f}, ensemble acc is {:.3f}'.format(acc1,acc2,acc3,acc4,acc))
		np.savetxt(os.path.join('exp','device1','epoch'+str(itr)+'acq'+str(acq)+'.txt'),AA1)
		np.savetxt(os.path.join('exp','device2','epoch'+str(itr)+'acq'+str(acq)+'.txt'),AA2)
		np.savetxt(os.path.join('exp','device3','epoch'+str(itr)+'acq'+str(acq)+'.txt'),AA3)
		np.savetxt(os.path.join('exp','device4','epoch'+str(itr)+'acq'+str(acq)+'.txt'),AA4)
		np.savetxt(os.path.join('exp','ensembled_'+'epoch'+str(itr)+'acq'+str(acq)+'.txt',),AA)
np.savetxt(os.path.join('exp','summary.txt'),recorder)
