from __future__ import division
from __future__ import print_function
import warnings
warnings.filterwarnings("ignore")

import time
import argparse
import pdb
import numpy as np
from skimage.morphology import disk
from skimage.morphology import binary_closing as closing
from skimage.measure import label
from skimage.io import imread, imsave
import matplotlib.pyplot as plt
from skimage.transform import resize
import glob

def largestCC(lImg,num=2):
	cIdx = np.zeros(num,dtype=int)
	count = np.bincount(lImg.flat)
	count[0] = 0 # Mark background count to zero
	lcc = np.zeros(lImg.shape,dtype=bool)
	if len(count) == 2:
		num = 1
	for i in range(num):
		cIdx[i] = np.argmax(count)
		count[cIdx[i]] = 0
		lcc += (lImg == cIdx[i])

	return lcc

def postProcess(img,s=11):
	#sEl = disk(args.disk)
#	plt.gray()
#	files = sorted(glob.glob(args.data+'*mask*.png'))
#	N = len(files)
#	for fIdx in range(N):
#	pdb.set_trace()
#	f = files[fIdx]
	bImg = (img > 0.5)
	if len(bImg.shape) > 2:
		bImg = bImg[:,:,-1]
	sEl = disk(s)
	lImg = label(bImg)
	lcc = largestCC(lImg) # Obtain the two largest connected components
	pImg = closing(lcc,sEl) 
	return pImg.astype(float)
