import os
from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import pylab
import pandas as pd
from datetime import datetime, time, timedelta
import tables
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import time, datetime, timedelta
from scipy.linalg import eigh, norm

#text imports
import gensim
from gensim.models import word2vec
import multiprocessing
cores = multiprocessing.cpu_count()
import random
assert gensim.models.word2vec.FAST_VERSION > -1, "this will be painfully slow otherwise"

#images imports
import tensorflow as tf
from scipy.misc import imread, imresize

#import classes
import common.p_vgg16 as p_vgg16
import common.p_cca as p_cca
import common.p_w2v as p_w2v

#get all usefull paths
MODEL_PATH = os.path.join('model')+os.sep
IMAGES_PATH = os.path.join('images', 'train2014')+os.sep

################################## CHOOSE SETTINGS ########################################
text_features = 'w2v' # can be set to 'GLOVE' to test the glove word representation
image_features = 'fc2' # choose which layer of vgg16 is going to be used to extract features
train_test_split = 50000
reg = 0.001 # this value is recommended by the paper
n_components = 50 # experiments have shown this value to be optimal
###########################################################################################


################################## LOAD MODELS ###########################################
#TEXT:
if text_features == 'GLOVE':
    #load glove dictionnary
    model = p_w2v.loadGloveModel('data/glove.6B/glove.6B.300d.txt')
else:
    #Load pre-trained word2vec NN, takes a bit of time
    model = gensim.models.Word2Vec.load_word2vec_format(MODEL_PATH+'GoogleNews-vectors-negative300.bin', binary=True)
# initialize COCO api for caption annotations
dataType='train2014'
annFile = 'annotations/captions_%s.json'%(dataType)
coco_caps=COCO(annFile)
imgIds = coco_caps.getImgIds()
catFile = 'annotations/instances_%s.json'%(dataType)
coco_insts = COCO(catFile)
catIds = coco_insts.getCatIds()

#Get Category retrieval dictionnary
dic_img_cat = {}
for imgId in imgIds:
    dic_img_cat[imgId] = []
for catId in catIds:
    imgIds_cat = coco_insts.getImgIds(catIds=catId)
    for imgId in imgIds_cat:
        dic_img_cat[imgId] += [coco_insts.loadCats(catId)[0]['name']]

#IMAGES
#initialize CNN
sess = tf.Session()
imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
vgg = p_vgg16.VGG16(imgs, MODEL_PATH + 'vgg16_weights.npz', sess)


################################## LOAD FEATURES ###########################################
hdf5_path = "data/data_"+image_features+"_all.hdf5"
read_hdf5_file = tables.open_file(hdf5_path, mode='r')
# Here we slice [:] all the data back into memory, then operate on it
text_data = read_hdf5_file.root.text[:]
image_data = read_hdf5_file.root.image[:]
cat_data = read_hdf5_file.root.cat[:]

read_hdf5_file.close()

if text_features == 'GLOVE':

    hdf5_path = "data/data_glove.hdf5"
    read_hdf5_file = tables.open_file(hdf5_path, mode='r')
    # Here we slice [:] all the data back into memory, then operate on it
    text_data = read_hdf5_file.root.text[:]
    read_hdf5_file.close()
###########################################################################################



if __name__ == '__main__':

    data = [text_data, image_data, cat_data]
    data_train = [text_data[train_test_split:], image_data[train_test_split:], cat_data[train_test_split:]]
    data_test = [text_data[:train_test_split], image_data[:train_test_split], cat_data[:train_test_split]]

    #initialize and train model
    cca = p_cca.CCA(model, coco_caps, vgg, sess, reg=reg, n_components=n_components)
    cca.train(data) # Here we train on all data in order to have the maximum number of images to query

    ############################## Tag to Image Search Example ###########################
    sentence = "dog dog skateboard"
    closest_ids = cca.T2I(sentence, neighbors_number=36)

    filename = 'images/results/6/'+sentence+'-'+str(image_features)+'-'+str(cca.reg)+'-'+str(cca.n_components)+'.png'
    result_images, result_captions = cca.getT2Iresults(closest_ids, filename, save = False, display = False)
    #######################################################################################

    ############################## Image to Tag Search Example ###########################
    id_query = imgIds.index(index_image)
    closest_ids = cca.I2T(id_query, image_features,IMAGES_PATH, neighbors_number = 100, cat = False, method = 'distance')
    #######################################################################################
