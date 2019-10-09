# CS 352 - Machine Learning
# Final Project
# Due: December 23rd, 2018
# Malcolm Gilbert and Weronika Nguyen

# This is a model for training and predicting a specific position of a LateX formula

import sys
import numpy as np
from skimage import io
from scipy.misc import imresize
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
import re
from utils import tokenize_formula, remove_invisible, normalize_formula

# Model architecture: https://github.com/tflearn/tflearn/blob/master/examples/images/convnet_mnist.py

tokens_size = 0
np.set_printoptions(threshold=np.inf)

class img2latexFirst:

    def __init__(self, targs, imgs, length=50, test_split=0.1, learning_rate=0.001):
        '''Initialize classes and image size.'''
        self.learning_rate = learning_rate
        self.image_data = []
        self.targets = targs
        self.images = imgs
        self.test_split = test_split
        self.voc_length = length

    def _process_image(self, image):

        '''Processing images - first load, crop and resize image. Add to the self.image_data array.'''
        # load in image
        old_img = io.imread('./train-data/'+image+'.png')
        img1 = old_img[:,:,3]
        img = img1.reshape([50, 60, 1])
        # add image to image_data
        self.image_data.append(np.array(img))

    def prepare_data(self):
        '''processes all the input images by calling _process_image'''
        for image in self.images:
            self._process_image(image)

    def process_test_img(self, image):
        '''processes a single testing image and returns it'''
        old_img = io.imread('./data/'+image+'.png')
        img = old_img[:,:,3]
        img = img.reshape([-1, 50, 60, 1])
        return img

    def build_model(self):
        convnet = input_data(shape=[None, 50, 60, 1], name='input')
        convnet = conv_2d(convnet, 32, 3, activation='relu', regularizer="L2")
        convnet = max_pool_2d(convnet, 2)
        convnet = local_response_normalization(convnet)
        convnet = conv_2d(convnet, 64, 3, activation='relu', regularizer="L2")
        convnet = max_pool_2d(convnet, 2)
        convnet = local_response_normalization(convnet)
        convnet = fully_connected(convnet, 128, activation='tanh')
        convnet = dropout(convnet, 0.8)
        convnet = fully_connected(convnet, 256, activation='tanh')
        convnet = dropout(convnet, 0.8)
        convnet = fully_connected(convnet, self.voc_length, activation='softmax')
        convnet = regression(convnet, optimizer='adam', learning_rate=0.001,
           loss='categorical_crossentropy', name='target')

        model = tflearn.DNN(convnet,
                            tensorboard_dir='log',
                            tensorboard_verbose=3
        )
        return model
        
    def train_model(self, model_name, epochs=5, batch_size=32):
        x = self.image_data
        y = self.targets
        X_train = np.asarray(x)
        y_train = np.asarray(y)
        indices = np.arange(X_train.shape[0])
        np.random.shuffle(indices)
        X_train = X_train[indices]
        y_train = y_train[indices]

        model = self.build_model()
        
        model.fit(X_train, y_train,
                  n_epoch=epochs,
                  shuffle=True,
                  validation_set=0.1,
                  show_metric=True,
                  batch_size=batch_size)
        
        self.model = model
        #self.model.save(model_name+".tfl")

    def load_model(self, model_file):
        model = self.build_model()
        model.load(model_file+".tfl")
        self.model = model

    def predict_formula(self, image, formula):
        # process testing img
        img = self.process_test_img(image)
        # get the arg max prediction
        old_result = self.model.predict(img)
        result = old_result.flatten()
        maxed = np.zeros(np.shape(result))
        max_val = max(result)
        index_of_max = np.nonzero(result == max_val)[0]
        maxed[index_of_max] = 1

        correct = "yes"
        for i in range(len(maxed)):
            if (maxed[i] != formula[i]):
                correct = "no"
        return maxed, formula, correct

if __name__ == '__main__':

    
    tok_formulas = {}         # full list of tokenized formulas
    unique_toks = set()       # a set containing unique tokens (it's going to be turned into a dictionary with indexes)
    form_to_indexes = {}      # full list of tokenized formulas in number (index) form for learning purposes
    max_len = 50              # maximum length of tokenized formulas
    token_dict = {}           # dictionary containg unique tokens

    f =  open('train-data.txt', "r", errors='ignore')
    formulas = f.readlines()
    f.close()

    f2 =  open('equations.txt', "r", errors='ignore')
    testing = f2.readlines()
    f2.close()

    
    # loop through each formula, remove invisible characters, normalize and turn into token list 
    for j in range(len(formulas)):
      i = remove_invisible(formulas[j])
      n = normalize_formula(i)
      t = tokenize_formula(n)

      tok_formulas[j] = t
      # add unique tokens to a set 
      for token in t:
        unique_toks.add(token)

    end_train = len(tok_formulas) # to get the last formula of train -> index=end-train-1

    for j in range(len(testing)):
      i = remove_invisible(testing[j])
      n = normalize_formula(i)
      t = tokenize_formula(n)

      tok_formulas[end_train+j] = t
      # add unique tokens to a set 
      for token in t:
        unique_toks.add(token)

    tokens_size = len(unique_toks)
    # turn the set containing unique tokens into a dictionary with tokes as keys and indexes (numbers) as values
    i=0
    for e in unique_toks:
      token_dict[e]=i
      i=i+1

    # turn each tokenized formula as a dictionary containing the indexes of the tokens instead of actual tokens
    for index in tok_formulas:
      new = []
      for token in tok_formulas.get(index):
        new.append(token_dict.get(token))
      form_to_indexes[index] = new
    

    hot_form_to_indexes = {}
    # turn targets into 1 hot vectors
    for i in range(len(form_to_indexes)):
      new = []
      k = form_to_indexes.get(i)
      for j in range(len(k)):
        row = np.zeros(len(unique_toks))
        c = k[j]
        row[c] = 1
        new.append(row)
      hot_form_to_indexes[i] = new

    train_targets = []
    train_images = []

    test_targets = []
    test_images = []

    # Initializing training images
    for i in range(end_train):
      train_targets.append(hot_form_to_indexes.get(i))
      train_images.append(str(i))

    # Initializing testting images
    j = 0
    for i in range(end_train, len(hot_form_to_indexes)):
      test_targets.append(hot_form_to_indexes.get(i))
      test_images.append(str(j))
      j=j+1 

    train_targets = np.array(train_targets)
    test_targets = np.array(test_targets)

    # To change the training position, overwrite the 3 in all the slicing "[:,3,:]" to a preferred index
    c = img2latexFirst(targs=train_targets[:,3,:], imgs=train_images, length=tokens_size, test_split=0.1, learning_rate=0.01)
    c.prepare_data()
    c.train_model(
        model_name="m",
        epochs=10,
        batch_size=100
        )
    # Get the prediction and accuracy percentage for exact match over all testing examples
    sum_correct=0
    for i in range(len(test_images)):
        r, f, a = c.predict_formula(test_images[i],test_targets[i][3,:])
        print("Prediction: ", r)
        print("Target: ", test_targets[i][3,:])
        if (a=="yes"):
            sum_correct = sum_correct+1

    print("Percent correct on all testing data:",sum_correct/len(test_images)*100)
  