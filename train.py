import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from data.dataset import lungData
import torch.nn as nn
from models.VAE import uVAE
import time
from utils.tools import makeLogFile, writeLog, dice_loss,dice,binary_accuracy
import pdb
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import argparse
torch.manual_seed(42)
np.random.seed(42)
from carbontracker.tracker import CarbonTracker
import os

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
parser.add_argument('--batch_size', type=int, default=12, help='Batch size')
parser.add_argument('--latent', type=int, default=8, help='Latent space dimension')
parser.add_argument('--beta', type=float, default=1.0, help='Scaling of KLD')
parser.add_argument('--hidden', type=int, default=16, help='Number of filters')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--data_path', type=str, default='/home/jwg356/covid/data/',help='Path to data.')
parser.add_argument('--aug',action='store_true', default=False,help='Use data aug.')
parser.add_argument('--blur',action='store_true', default=False,help='Use blurry masks.')
parser.add_argument('--block',action='store_true', default=False,help='Use block masks')
parser.add_argument('--unet',action='store_true', default=False,help='Use only U-Net.')

fName = time.strftime("%Y%m%d_%H_%M")
logFile = '../logs/'+fName+'.txt'

args = parser.parse_args()
if args.unet:
	print("Using U-Net without VAE")
	fName = fName+'_unet'
else:
	print("Using U-Net + VAE")
	fName = fName+'_vae'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


### Choose augmentation method
if args.aug:
	dataset = lungData(data_dir=args.data_path,blur=args.blur,
					block=args.block,hflip=False,vflip=False,rot=15,p=0.1,rMask=50)
	print("Using data augmentation....")
	if args.block:
		print("Using block masks in inputs")
		fName = fName+'_block'
	elif args.blur:
		print("Using diffuse noise in inputs")
		fName = fName+'_diff'
	else:
		print("Using both masks in inputs")
		fName = fName+'_both'
else:
	dataset = lungData(data_dir=args.data_path,
						hflip=True,vflip=True,rot=15,p=0.1,rMask=0)
	print("Standard augmentation...")
	fName = fName+'_noAug'

fName = fName+'_'+repr(args.hidden)+'_hid'

# Location to save validation visualization at each epoch
if not os.path.exists('../vis/'+fName):
	os.mkdir('../vis/'+fName)
# Location to save current best model
if not os.path.exists('saved_models/'+fName):
	os.mkdir('../saved_models/'+fName)

dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(0.25 * dataset_size))

np.random.shuffle(indices)
valid_indices, train_indices = \
				indices[:split], indices[split:]

train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(valid_indices)

train_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=train_sampler)
valid_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=valid_sampler)

print("Number of train/valid patches:", 
				(len(train_indices),len(valid_indices)))

net = uVAE(nhid=args.hidden,nlatent=args.latent,unet=args.unet)
nParam = sum(p.numel() for p in net.parameters() if p.requires_grad)
print("Model"+fName+"Number of parameters:%d"%(nParam))
with open(logFile,"a") as f:
    print("Model:"+fName+"Number of parameters:%d"%(nParam),file=f)
makeLogFile(logFile)

criterion = nn.BCELoss(reduction='mean')
accuracy = dice

net.to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-5)
nTrain = len(train_loader)
nValid = len(valid_loader)

minLoss = 1e5
convIter=0
convCheck = 20
tracker = CarbonTracker(epochs=args.epochs,log_dir='../logs/',monitor_epochs=-1)
beta = args.beta

for epoch in range(args.epochs):
	tracker.epoch_start()
	trLoss = []
	vlLoss = []
	trAcc = []
	vlAcc = []
	t = time.time()

	for step, (patch, mask) in enumerate(train_loader): 
		patch = patch.to(device)
		mask = mask.to(device)
		kl, pred = net(patch)
		pred = torch.sigmoid(pred)
		rec_loss = criterion(target=mask,input=pred)
		loss = beta*kl/mask.shape[0] + rec_loss
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		trAcc.append(kl.item()/mask.shape[0])
		trLoss.append(loss.item())
		if (step+1) % 5 == 0:
			with torch.no_grad():
				for idx, (patch, mask) in enumerate(valid_loader): 
					patch = patch.to(device)
					mask = mask.to(device)
					kl, pred = net(patch)
					pred = torch.sigmoid(pred)
					rec_loss = criterion(target=mask, input=pred)
					loss = beta*kl/mask.shape[0] + rec_loss
					vlLoss.append(loss.item())
					vlAcc.append(kl.item()/mask.shape[0])
					break
				print ('Epoch [{}/{}], Step [{}/{}], TrLoss: {:.4f}, VlLoss: {:.4f}'
					.format(epoch+1, args.epochs, step+1, 
							nTrain, trLoss[-1], vlLoss[-1]))
	epValidLoss =  np.mean(vlLoss)
	epValidAcc = np.mean(vlAcc)
	if (epoch+1) % 1 == 0 and epValidLoss < minLoss:
		convIter = 0
		minLoss = epValidLoss
		print("New max: %.4f\nSaving model..."%(minLoss))
		torch.save(net.state_dict(),'../saved_models/'+fName+'/epoch_%03d.pt'%(epoch+1))
		img = torch.zeros((2*mask.shape[0],3,mask.shape[2],mask.shape[3]))
		img[::2] = patch
		pred =  ((pred.detach() > 0.5).float() + 2*mask).squeeze()
		img[1::2,0][pred == 1] = 0.55
		img[1::2,2][pred == 2] = 0.6
		img[1::2,1][pred == 3] = 0.75
		save_image(img,'../vis/'+fName+'/epoch_%03d.jpg'%(epoch+1))
	else:
		convIter += 1
	writeLog(logFile, epoch, np.mean(trLoss), np.mean(trAcc),
					   epValidLoss,np.mean(vlAcc), time.time()-t)
	tracker.epoch_end()
	if convIter == convCheck:
		print("Converged at epoch %d"%(epoch+1-convCheck))
		break
	elif np.isnan(epValidLoss):
		print("Nan error!")
		break
tracker.stop()
