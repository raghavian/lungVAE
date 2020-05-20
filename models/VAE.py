import torch
import torch.nn as nn
from torch.distributions import Normal, Independent
from torch.distributions.kl import kl_divergence as KLD
import numpy as np
from torch.nn.functional import softplus, sigmoid, softmax
import pdb
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class convBlock(nn.Module):
	def __init__(self, inCh, nhid, nOp, pool=True,
					ker=3,padding=1,pooling=2):
		super(convBlock,self).__init__()

		self.enc1 = nn.Conv2d(inCh,nhid,kernel_size=ker,padding=1)
		self.enc2 = nn.Conv2d(nhid,nOp,kernel_size=ker,padding=1)
		self.bn = nn.BatchNorm2d(inCh)	

		if pool:
			self.scale = nn.AvgPool2d(kernel_size=pooling)
		else:
			self.scale = nn.Upsample(scale_factor=pooling)
		self.pool = pool
		self.act = nn.ReLU()

	def forward(self,x):
		x = self.scale(x)
		x = self.bn(x)
		x = self.act(self.enc1(x))
		x = self.act(self.enc2(x))
		return x

class uVAE(nn.Module):
	def __init__(self, nlatent,unet=False, 
					nhid=8, ker=3, inCh=1,h=640,w=512):
		super(uVAE, self).__init__()
		self.latent_space = nlatent
		self.unet = unet

		if not self.unet:
			### VAE Encoder with 3 downsampling layers
			self.enc11 = nn.Conv2d(inCh,nhid,kernel_size=ker,padding=1)
			self.enc12 = nn.Conv2d(nhid,nhid,kernel_size=ker,padding=1)

			self.enc2 = convBlock(nhid,2*nhid,2*nhid,pool=True)
			self.enc3 = convBlock(2*nhid,4*nhid,4*nhid,pool=True)
			self.enc4 = convBlock(4*nhid,8*nhid,8*nhid,pool=True)
			self.enc5 = convBlock(8*nhid,16*nhid,16*nhid,pool=True)

			self.bot11 = nn.Conv1d(16*nhid,1,kernel_size=1)
			self.bot12 = nn.Conv1d(int((h/16)*(w/16)),2*nlatent,kernel_size=1)

			### Decoder with 3 upsampling layers
			self.bot21 = nn.Conv1d(nlatent,int((h/64)*(w/64)),kernel_size=1)
			self.bot22 = nn.Conv1d(1,nhid,kernel_size=1)
			self.bot23 = nn.Conv1d(nhid,4*nhid,kernel_size=1)
			self.bot24 = nn.Conv1d(4*nhid,16*nhid,kernel_size=1)

		### U-net Encoder with 3 downsampling layers
		self.uEnc11 = nn.Conv2d(inCh,nhid,kernel_size=ker,padding=1)
		self.uEnc12 = nn.Conv2d(nhid,nhid,kernel_size=ker,padding=1)

		self.uEnc2 = convBlock(nhid,2*nhid,2*nhid,pool=True,pooling=4)
		self.uEnc3 = convBlock(2*nhid,4*nhid,4*nhid,pool=True,pooling=4)
		self.uEnc4 = convBlock(4*nhid,8*nhid,8*nhid,pool=True)
		self.uEnc5 = convBlock(8*nhid,16*nhid,16*nhid,pool=True)

		### Joint U-Net + VAE decoder 
		if not self.unet:
			self.dec5 = convBlock(32*nhid,8*nhid,8*nhid,pool=False)
		else:
			self.dec5 = convBlock(16*nhid,8*nhid,8*nhid,pool=False)

		self.dec4 = convBlock(16*nhid,4*nhid,4*nhid,pool=False)
		self.dec3 = convBlock(8*nhid,2*nhid,2*nhid,pool=False,pooling=4)
		self.dec2 = convBlock(4*nhid,nhid,nhid,pool=False,pooling=4)

		self.dec11 = nn.Conv2d(2*nhid,nhid,kernel_size=ker,padding=1)
		self.dec12 = nn.Conv2d(nhid,inCh,kernel_size=ker,padding=1)
		
		self.act = nn.ReLU()
		self.mu_0 = torch.zeros((1,nlatent)).to(device)
		self.sigma_0 = torch.ones((1,nlatent)).to(device)

		self.h = h
		self.w = w

	def vae_encoder(self,x):
		### VAE Encoder
		x = self.act(self.enc11(x))
		x = self.act(self.enc12(x))
		x = self.enc2(x)
		x = self.enc3(x)
		x = self.enc4(x)
		x = self.enc5(x)

		z = self.act(self.bot11(x.view(x.shape[0],x.shape[1],-1)))
		z = self.bot12(z.permute(0,2,1))

		return z.squeeze(-1)

	
	def unet_encoder(self,x_in):
		### VAE Encoder
		x = []
		
		x.append(self.act(self.uEnc12(self.act(self.uEnc11(x_in)))))
		x.append(self.uEnc2(x[-1]))
		x.append(self.uEnc3(x[-1]))
		x.append(self.uEnc4(x[-1]))
		x.append(self.uEnc5(x[-1]))

		return x

	def decoder(self,x_enc,z=None):
		if not self.unet:
				### Concatenate latent vector to U-net bottleneck
				x = self.act(self.bot21(z.unsqueeze(2)))
				x = self.act(self.bot22(x.permute(0,2,1)))
				x = self.act(self.bot23(x))
				x = self.act(self.bot24(x))

				x = x.view(x.shape[0],x.shape[1],
						int(self.h/64),int(self.w/64))
				x = torch.cat((x,x_enc[-1]),dim=1)
				x = self.dec5(x)
		else:
				x = self.dec5(x_enc[-1])
		
		### Shared decoder
		x = torch.cat((x,x_enc[-2]),dim=1)
		x = self.dec4(x)
		x = torch.cat((x,x_enc[-3]),dim=1)
		x = self.dec3(x)
		x = torch.cat((x,x_enc[-4]),dim=1)
		x = self.dec2(x)
		x = torch.cat((x,x_enc[-5]),dim=1)

		x = self.act(self.dec11(x))
		x = self.dec12(x)

		return x

	def forward(self, x):
		kl = torch.zeros(1).to(device)
		z = 0.
		# Unet encoder result
		x_enc = self.unet_encoder(x)

		# VAE regularisation
		if not self.unet:
				emb = self.vae_encoder(x)

				# Split encoder outputs into a mean and variance vector
				mu, log_var = torch.chunk(emb, 2, dim=1)

				# Make sure that the log variance is positive
				log_var = softplus(log_var)
				sigma = torch.exp(log_var / 2)
				
				# Instantiate a diagonal Gaussian with mean=mu, std=sigma
				# This is the approximate latent distribution q(z|x)
				posterior = Independent(Normal(loc=mu,scale=sigma),1)
				z = posterior.rsample()

				# Instantiate a standard Gaussian with mean=mu_0, std=sigma_0
				# This is the prior distribution p(z)
				prior = Independent(Normal(loc=self.mu_0,scale=self.sigma_0),1)

				# Estimate the KLD between q(z|x)|| p(z)
				kl = KLD(posterior,prior).sum() 	

		# Outputs for MSE
		xHat = self.decoder(x_enc,z)

		return kl, xHat


