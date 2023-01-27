# %matplotlib notebook

### interactive notebook format is required for the interactive plot

import numpy as np
import pandas as pd
import math
from math import isnan
import random
import os
from os import listdir
from os.path import isfile, join
import cv2
from skimage.transform import resize
import csv
from functools import partial
from tqdm import tqdm

from sklearn.decomposition import PCA
from sklearn.manifold import MDS
import sklearn.metrics.pairwise
from sklearn.metrics import silhouette_score

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.patches import FancyBboxPatch
from matplotlib.widgets import Slider, Button

import ipywidgets as widgets
from ipywidgets import interact, Layout, Button, GridBox, ButtonStyle
from IPython.display import display, clear_output, Image

import ipyplot

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
from torchvision import transforms
from torchvision.models import resnet18, resnet34, resnet101, resnet152

import copy
from PIL import Image

## see VisualBackProp.py
import VisualBackProp as VBP


## settings
imgFolder = '/users/huiminhan/Desktop/Lab/InfoVis/Datasets/animal_sample/'
# imgFolder = '/users/huiminhan/Desktop/Lab/InfoVis/Datasets/pods/3cat/'
# imgFolder = '/users/huiminhan/Desktop/Lab/InfoVis/Datasets/pokemon/'
sampleSizePerCat = 200  #sample size for each image category subfolder
imgDisplaySize = 0.2 #default value for image display size, can be interactively adjusted in the UI
total_img = 800  # maximun total number of images to display on the UI
folderName = True  #whether the imgFolder has subfolder for each category. e.g. the fish dataset
load_weights_from_file = False

class FilenameDataset(Dataset):

    def __init__(self, files, transform=None):
        self.files = list(files)
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        sample = Image.open(self.files[idx]).convert('RGB')
        if self.transform: #whether self-defined transform
            return self.transform(sample)
        transform_default = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        return transform_default(sample)
    

def get_path(imgFolder, sampleSizePerCat, folderName, max_img_num=50):
    """    
    @parameters:
        imgFolder[str]: path for the image folder. e.g. "/root/" 
        sampleSizePerCat[int]: the number of samples from each subfolder. e.g. 10 images each from "root/cat/"
                          and "root/dog/"
        folderName[boolean]: if structure is: root/cat/123.png then True, if root/123.png then False.
        max_img_num[int]: upper limit for TOTAL image samples
    @return[dict]: a dictionary with image index (used in later dataframe as index as well)
             as key and full path of the image as the value
    """
    imgIdx_path = {}
    totalImg = 0
    for (dirpath, dirnames, filenames) in os.walk(imgFolder):
        sampleCount = 0
        for filename in filenames:
            if filename.lower().endswith('jpg') or filename.lower().endswith(
                    'jpeg') or filename.lower().endswith('png'):
                path = dirpath + '/' + filename
                pattern = extractIdx_pattern(path, folderName)
                imgIdx_path[pattern] = path
                sampleCount += 1
                totalImg += 1
            if sampleCount == sampleSizePerCat or totalImg == max_img_num:
                break
        if totalImg == max_img_num:
            break
    return imgIdx_path


def extractIdx_pattern(path, folderName):
    """
    @parameters:
        path[str]: single image path. e.g. "/root/cat/cat_01.png"
        folderName[boolean]: if structure is: root/cat/123.png then True, if root/123.png then False.
    @return[sts]: given a path(string) of image, extract image index from the path, return the image index string
    """
    if folderName:
        pattern = path.split('/')[-2] + '/' + path.split('/')[-1].split(
            '.')[-2]
    else:
        pattern = path.split('/')[-1].split('.')[-2]
    return pattern


def data_loader(imgIdx_path):
    """
    @parameters:
        imgIdx_path[dict]: a dict get from get_path function storing {image index: full path of the image}
    @return: image loader
    """

    dataset = FilenameDataset(imgIdx_path.values())
    loader = DataLoader(dataset)
    if loader:
        print("{} images loaded".format(len(loader)))
        return loader
    else:
        print("Invalid path")
        return


def feature_extractor(model, loader, imgIdx_path):
    """
    @parameters:
        model[neuron network]: model used to extract features
        loader: image loader returned from the data_loader function
    @return[dataframe]: a dataframe of extracted features indexing by image index(get from extractIdx_pattern function)
    """
    features = []
    for i, img in zip(range(len(loader)), loader):
        with torch.no_grad():
            x, vis, target_feature_map = model(img)
            features.append(x)
        index = []
        for path in imgIdx_path.values():
            index.append(extractIdx_pattern(path, folderName))
    df = pd.DataFrame(features,
                      columns=[str(i) for i in range(1, 513)],
                      index=index)
    df.index.name = 'Image'
    return df


def df_preprocess(df_image, normalize=True):
    """
    @parameters:
        df_image[dataframe]: image features dataframe
    @return[dataframe]: preprocessed dataframe
    """
    df_image.sort_index(inplace=True)
    df_numeric = df_image.select_dtypes(include='number').drop_duplicates(
    )  #'int32' or 'int64' or 'float32' or 'float64'
    df_category = df_image.select_dtypes(
        exclude='number').drop_duplicates()  #'object'
    ### Z-score normalization
    if normalize:
        normalized_df = (df_numeric - df_numeric.mean()) / df_numeric.std(
        )  # do not normalize animal dataset, all columns are 0-100 scale
        return normalized_df
    return df_numeric

imageIndex_path_dict = get_path(imgFolder,
                                sampleSizePerCat,
                                folderName=folderName,
                                max_img_num=total_img)
img_loader = data_loader(imageIndex_path_dict)

## using the same model in backprop file to make sure
## the forward and backward process happens to the same network
model = resnet18(pretrained=True).eval()
model_bp = VBP.ResnetVisualizer(model.eval(), weight_list=torch.ones([512]))
df = feature_extractor(model_bp, img_loader, imageIndex_path_dict)
#################################
# the name of the dataframe must be normalized_df, used as global variable later
normalized_df = df_preprocess(df)

print(normalized_df)