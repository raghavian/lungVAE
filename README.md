# README #

This is official Pytorch implementation of Lung Segmentation network with Variational data imputation 
"[Lung Segmentation from Chest X-rays using Variational Data Imputation](https://arxiv.org/abs/2004.10076)", Raghavendra Selvan et al. 2020

![lotenet](models/model.png)
### What is this repository for? ###

* Predict lung masks from CXRs
* Run and reproduce results in the paper
* v1.0

### How do I get set up? ###

* Basic Pytorch dependency
* Tested on Pytorch 1.3, Python 3.6 
* How to run tests: python train.py --data data_location
* How to predict: python predict.py --data input_location --post 
### Usage guidelines ###

* Kindly cite our publication if you use any part of the code

@article{raghav2020lungVAE,
 	title={Lung Segmentation from Chest X-rays using Variational Data Imputation},
	author={Raghavendra Selvan et al.},
 	journal={arXiv preprint arXiv:2020.00000,
	year={2020}
}
``
### Who do I talk to? ###

* raghav@di.ku.dk
