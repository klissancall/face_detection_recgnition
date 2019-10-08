import numpy as np
import time
import cv2
from boosting_classifier import Boosting_Classifier
from visualizer import Visualizer
from im_process import normalize
from utils import *


#flag for debugging
flag_subset = False
boosting_type = 'Real' #'Real' or 'Ada'
training_epochs = 100 if not flag_subset else 20
act_cache_dir = 'wc_activations.npy' if not flag_subset else 'wc_activations_subset.npy'
chosen_wc_cache_dir = 'chosen_wcs.pkl' if not flag_subset else 'chosen_wcs_subset.pkl'
chosen_wc_cache_dir_real =  'chosen_wcs_real.pkl' if not flag_subset else 'chosen_wcs_subset_real.pkl'

#data configurations
pos_data_dir = 'newface16'
neg_data_dir = 'nonface16'
image_w = 16
image_h = 16
data, labels = load_data(pos_data_dir, neg_data_dir, image_w, image_h, flag_subset)
data = integrate_images(normalize(data))

#number of bins for boosting
num_bins = 25

#number of cpus for parallel computing
num_cores = 1 if not flag_subset else 1 #always use 1 when debugging

#create Haar filters
filters = generate_Haar_filters(4, 4, 16, 16, image_w, image_h, flag_subset)

#create visualizer to draw histograms, roc curves and best weak classifier accuracies
drawer = Visualizer([10, 20, 50, 100], [1, 10, 20, 50, 100])

#Adaboost
real = Boosting_Classifier(filters, data, labels, training_epochs, num_bins, drawer, num_cores, boosting_type)

#calculate filter values for all training images
start = time.clock()
real.calculate_training_activations(act_cache_dir, act_cache_dir)
end = time.clock()
#print('%f seconds for activation calculation' % (end - start))

real.load_trained_wcs('chosen_wcs.pkl')

real.train(chosen_wc_cache_dir_real)

#st = 1
#ii = cv2.imread('./Test Images 2018/Hard Negative 1.jpeg', cv2.IMREAD_GRAYSCALE)

#wrong_img = boost.get_hard_negative_patches(ii,scale_step=st)
#print(wrong_img.shape)
