#GATLGEMF：Predicting ncRNA-protein interactions based on line graph attention networks

The _untils_, _data_ and _result_ directories contain model codes, data sets and generated results, respectively.
The depended python packages are listed in requirements.txt and environments.yml. The package versions should be followed by users in their environments to achieve the supposed performance.

# Dependency:

python 3.6.13

#### Python Packages
torch==1.10.1+cu113
torch-geometric==2.0.3
matplotlib==3.2.2
networkx==2.4
numpy==1.19.5
pandas==1.1.5
scipy==1.5.4
scikit-learn==0.24.2

# NOTE

Before using Main.py, if you want to use node feature, you need to use the below bash command to obtain node feature.

```bash
    python node_feature.py --dataset DATASET
```

#Train the network
The program is in Python 3.6.13 using [Pytorch] backends. Use the below bash command to run NPI-LGAT.

```bash
    python Main.py --dataset DATASET
```

The parameter of DATASET could be RPI369, RPI2241, NPInter4158，RPI7317 and NPInter v2.0. Then, GATLGEMF will perform 5-fold cross validation on the specific dataset.

#Indenpendent test
We demonstrated the model in jupyter notebook. The independent_test.ipynb is in the script folder. The test data and test model are in the data folder.


# Reference:
(If using this code , please cite our paper.)
GATLGEMF：Predicting ncRNA-protein interactions based on line graph attention networks

# Contact: tanjianjun@bjut.edu.cn


