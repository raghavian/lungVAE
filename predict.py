import torch

import numpy as np
import torch.nn as nn
from models.VAE import uVAE
import time
from glob import glob
import pdb
import argparse
torch.manual_seed(42)
np.random.seed(42)
from pydicom import dcmread
from skimage.transform import resize
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
from skimage.exposure import equalize_hist as equalize
from skimage.io import imread,imsave
from skimage import img_as_ubyte
from skimage.color import rgb2gray
from utils.postProcess import postProcess
from utils.tools import dice,binary_accuracy
from torchvision.utils import save_image
import os
plt.gray()
	
def loadDCM(f, no_preprocess=False,dicom=False):
	wLoc = 448
	### Load input dicom
	if dicom:
		dcmFile = dcmread(f)
		dcm = dcmFile.pixel_array
		dcm = dcm/dcm.max()
		if dcmFile.PhotometricInterpretation == 'MONOCHROME1':
			### https://dicom.innolitics.com/ciods/ct-image/image-pixel/00280004 ###
			### When MONOCHROME1, 0->bright, 1->dark intensities
			dcm = 1-dcm 
	else:
		## Load input image
		dcm = imread(f)
		dcm = dcm/dcm.max()
	if not no_preprocess:
		dcm = equalize(dcm)

	if len(dcm.shape) > 2:
		dcm = rgb2gray(dcm[:,:,:3])
	
	### Crop and resize image to 640x512 
	hLoc = int((dcm.shape[0]/(dcm.shape[1]/wLoc)))
	if hLoc > 576:
		hLoc = 576
		wLoc = int((dcm.shape[1]/(dcm.shape[0]/hLoc)))

	img = resize(dcm,(hLoc,wLoc))
	img = torch.Tensor(img)
	pImg = torch.zeros((640,512))
	h = (int((576-hLoc)/2))+p
	w = int((448-wLoc)/2)+p
	roi = torch.zeros(pImg.shape)
	if w == p:
		pImg[np.abs(h):(h+img.shape[0]),p:-p] = img
		roi[np.abs(h):(h+img.shape[0]),p:-p] = 1.0
	else:
		pImg[p:-p,np.abs(w):(w+img.shape[1])] = img	
		roi[p:-p,np.abs(w):(w+img.shape[1])] = 1.0

	imH = dcm.shape[0]
	imW = dcm.shape[1]
	pImg = pImg.unsqueeze(0).unsqueeze(0)
	return pImg, roi, h, w, hLoc, wLoc, imH, imW

def saveMask(f,img,h,w,hLoc,wLoc,imH,imgW,no_post=False):
	
	img = img.data.numpy()
	imgIp = img.copy()
	
	if w == p:
		img = resize(img[np.abs(h):(h+hLoc),p:-p],
					(imH,imW),preserve_range=True)
	else:
		img = resize(img[p:-p,np.abs(w):(w+wLoc)],
					(imH,imW),preserve_range=True)
	imsave(f,img_as_ubyte(img>0.5))

	if not no_post:
		imgPost = postProcess(imgIp)
		if w == p:
			imgPost = resize(imgPost[np.abs(h):(h+hLoc),p:-p],
							(imH,imW))
		else:
			imgPost = resize(imgPost[p:-p,np.abs(w):(w+wLoc)],
							(imH,imW))

		imsave(f.replace('.png','_post.png'),img_as_ubyte(imgPost > 0.5))

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='.', help='Path to input files')
parser.add_argument('--model', type=str, default='saved_models/lungVAE.pt', help='Path to trained model')
parser.add_argument('--hidden', type=int, default=16, help='Hidden units')
parser.add_argument('--latent', type=int, default=8, help='Latent dim')
parser.add_argument('--saveLoc', type=str, default='', help='Path to save predictions')
parser.add_argument('--unet',action='store_true', default=False,help='Use only U-Net.')
parser.add_argument('--dicom',action='store_true', default=False,help='DICOM inputs.')
parser.add_argument('--no_post',action='store_true', default=False,help='Do not post process predictions')
parser.add_argument('--no_preprocess',action='store_true', 
						default=False,help='Do not preprocess input images')
parser.add_argument('--padding', type=int, default=32, help='Zero padding')


args = parser.parse_args()
p = args.padding
print("Loading "+args.model)
if 'unet' in args.model:
	args.unet = True
	args.hidden = int(1.5*args.hidden)
else:
	args.unet = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = uVAE(nhid=args.hidden,nlatent=args.latent,unet=args.unet)
net.load_state_dict(torch.load(args.model,map_location=device))
net.to(device)
t = time.strftime("%Y%m%d_%H_%M")

if args.saveLoc is '':
	save_dir = args.data+'pred_'+t+'/'
else:
	save_dir = args.saveLoc+'pred_'+t+'/'
if not os.path.exists(save_dir):
	os.mkdir(save_dir)

nParam = sum(p.numel() for p in net.parameters() if p.requires_grad)
print("Model "+args.model.split('/')[-1]+" Number of parameters:%d"%(nParam))

if args.dicom:
	filetype = 'DCM'
else:
	filetype= 'png'

files = list(set(glob(args.data+'*.'+filetype)) \
			- set(glob(args.data+'*_mask*.'+filetype)) \
			- set(glob(args.data+'*label*.'+filetype)))

files = sorted(files)
for fIdx in range(len(files)):
		f = files[fIdx]
		fName = f.split('/')[-1]
		img, roi, h, w, hLoc, wLoc, imH, imW = loadDCM(f,
													no_preprocess=args.no_preprocess,
													dicom=args.dicom)
		img = img.to(device)
		_,mask = net(img)
		mask = torch.sigmoid(mask*roi)
		f = save_dir+fName.replace('.'+filetype,'_mask.png')

		saveMask(f,mask.squeeze(),h,w,hLoc,wLoc,imH,imW,args.no_post)
		print("Segmenting %d/%d"%(fIdx,len(files)))
