import os
#import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader, Dataset
from PIL import Image
import pdb
from numpy import random
import glob
from skimage.transform import resize
import numpy as np
from skimage.exposure import equalize_hist as equalize
from skimage.draw import random_shapes
from skimage.filters import gaussian

def make_masks(h,w,s,N,blur=False):
	masks = np.zeros((N,1,h,w))
	for i in range(N):
		skMask =  (random_shapes((h,w),min_shapes=10,max_shapes=20,
					min_size=s,allow_overlap=True,
					multichannel=False,shape='circle')[0] < 128).astype(float)
		if blur:
			skMask = gaussian(skMask,sigma=s/2) # 0.8 is ~dense opacity
		masks[i,0] = skMask

	pdb.set_trace()
	if blur:
		masks = torch.Tensor(masks) 
	else:
		masks = torch.Tensor(masks).dtype(torch.BoolTensor)
	return masks
	
	
def pad(data):
	
	pImg = torch.zeros((1,640,512))
	h = (int((640-data.shape[1])/2) )
	w = int((512-data.shape[2])/2)  
	if w == 0: 
		pImg[0,np.abs(h):(h+data.shape[1]),:] = (data[0]) 
	else: 
		pImg[0,:,np.abs(w):(w+data.shape[2])] = (data[0])
	return pImg

class lungData(Dataset):
	def __init__(self,data_dir = '/home/jwg356/covid/data/', process=False,
					hflip=False,vflip=False,rot=0,p=0.5,rMask=0,block=False,
					blur=True,transform=None):

		super().__init__()
		self.h = 640
		self.w = 512
		self.data_dir = data_dir
		self.hflip = hflip
		self.vflip = vflip
		self.rot = rot
		self.rMask = rMask
		self.blur = blur  # Use blurry masks
		self.block = block
#		pdb.set_trace()
		if process:
			self.process()
		self.data, self.targets = torch.load(data_dir+'processed/lungData.pt')
		self.p = p
		if self.rMask > 0:
			if not (self.block):
				self.masks = torch.load(data_dir+'processed/blurry_masks.pt')

	def __len__(self):
		return len(self.targets)

	def __getitem__(self, index):

		image, label = self.data[index], self.targets[index]
		image, label = TF.to_pil_image(image), TF.to_pil_image(label)

		do_hflip = random.random() > self.p
		do_vflip = random.random() > self.p
		do_rot   = random.random()
		do_rMask = random.random()
		
		if self.hflip and do_hflip:
			image,label = TF.hflip(image), TF.hflip(label)

		if self.vflip and do_vflip:
			image,label = TF.vflip(image), TF.vflip(label)

		if (self.rot > 0) and (do_rot > self.p):
			if do_rot >= (1-self.p)/2:
				angle = -self.rot * do_rot
			else:
				angle = self.rot * do_rot
			image,label = TF.rotate(image,angle),\
							TF.rotate(label,angle)

		image, label = TF.to_tensor(image), TF.to_tensor(label)
		if (self.rMask > 0) and (do_rMask > self.p):
				if self.blur:
					idx = random.randint(self.masks.shape[0])
					mask = self.masks[idx]
					image += (0.6*mask-0.1)
				elif self.block:
					if do_rMask >= 3*(1-self.p)/4:
						image[0,320:,:] = 0.85
					elif do_rMask >= 2*(1-self.p)/4:
						image[0,:,:256] = 0.85
					elif do_rMask >= (1-self.p)/4:
						image[0,:,256:] = 0.85
					else:
						image[0,:320,:] = 0.85
				else:
					idx = random.randint(self.masks.shape[0])
					mask = self.masks[idx]
					if do_rMask >= (1-self.p)/2:
						image += (0.6*mask-0.1)
					elif do_rMask >= 3*(1-self.p)/8:
						image[0,320:,:] = 0.85
					elif do_rMask >= 2*(1-self.p)/8:
						image[0,:,:256] = 0.85
					elif do_rMask >= (1-self.p)/8:
						image[0,:,256:] = 0.85
					else:
						image[0,:320,:] = 0.85

		return image, label

	def process(self):
			mask = sorted(glob.glob(self.data_dir+'raw/masks/*.png'))
			data = [f.split('/')[-1].replace('_mask.png','.png') for f in mask]
			N = len(mask)
			images = torch.zeros((N,1,self.h,self.w))
			labels = torch.zeros((N,1,self.h,self.w))
			for index in range(N):
				image = Image.open(self.data_dir+'raw/equalized/'+data[index])
				label = Image.open(mask[index])
				h = int(image.height/(image.width/self.w))
				if h > self.h:
					self.w = int(image.width/(image.height/self.h))
				image, label = TF.resize(image,(self.h,self.w)), \
								TF.resize(label,(self.h,self.w))
				image, label = TF.to_tensor(image), TF.to_tensor(label)

				image, label = pad(image), pad(label)
				images[index],labels[index] = image, label

			masks = make_masks(self.h,self.w,self.rMask,int(2*N))
			torch.save(masks,self.data_dir+'processed/random_masks.pt')
			torch.save((images,labels),self.data_dir+'processed/lungData.pt')




