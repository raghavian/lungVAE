import pydicom
from skimage.transform import resize
import os,pdb
import glob
import matplotlib.pyplot as plt

path='/home/jwg356/erda/COVID-19/DK_images/'
os.chdir(path)
files = sorted(glob.glob('*.DCM'))
pdb.set_trace()

for f in files:
		dcm = pydicom.dcmread(f).pixel_array
		plt.imsave('../workDir/'+f.replace('.DCM','.png'),dcm)
	

