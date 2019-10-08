import numpy as np
import time
import cv2
from boosting_classifier import Boosting_Classifier
from visualizer import Visualizer
from im_process import normalize
from utils import *

def main():
    # flag for debugging
    flag_subset = False
    boosting_type = 'Ada'  # 'Real' or 'Ada'
    training_epochs = 100 if not flag_subset else 20
    act_cache_dir = 'wc_activations.npy' if not flag_subset else 'wc_activations_subset.npy'
    chosen_wc_cache_dir = 'chosen_wcs.pkl' if not flag_subset else 'chosen_wcs_subset.pkl'
    #chosen_wc_cache_dir_real = 'chosen_wcs_real.pkl' if not flag_subset else 'chosen_wcs_subset_real.pkl'

    # data configurations
    pos_data_dir = 'newface16'
    neg_data_dir = 'nonface16'
    image_w = 16
    image_h = 16
    data, labels = load_data(pos_data_dir, neg_data_dir, image_w, image_h, flag_subset)
    data = integrate_images(normalize(data))

    # number of bins for boosting
    num_bins = 25

    # number of cpus for parallel computing
    num_cores = 1 if not flag_subset else 1  # always use 1 when debugging

    # create Haar filters
    filters = generate_Haar_filters(4, 4, 16, 16, image_w, image_h, flag_subset)

    # create visualizer to draw histograms, roc curves and best weak classifier accuracies
    drawer = Visualizer([10, 20, 50, 100], [1, 10, 20, 50, 100])

    # Adaboost
    boost = Boosting_Classifier(filters, data, labels, training_epochs, num_bins, drawer, num_cores, boosting_type)

    # calculate filter values for all training images
    #start = time.clock()
    #boost.calculate_training_activations(act_cache_dir, act_cache_dir)
    #end = time.clock()
    # print('%f seconds for activation calculation' % (end - start))
    #boost.train(chosen_wc_cache_dir)
    #boost.load_trained_wcs('chosen_wcs_plus.pkl')
    boost.load_trained_wcs('chosen_wcs.pkl')
    for i in range(20):
        id = boost.chosen_wcs[i][1].id
        print("Filter ID = "+str(id)+" Alpha = "+str(boost.chosen_wcs[i][0]))
        boost.draw_filter(filters[id],id,boost.chosen_wcs[i][1].polarity)

    # Test
    st = 10
    original_img = cv2.imread('./test/Face_1.jpg', cv2.IMREAD_GRAYSCALE)
    result_img = boost.face_detection(original_img, scale_step=st)
    cv2.imwrite('Result_img_No_1_%s_plus.png' % boosting_type, result_img)
    original_img = cv2.imread('./test/Face_2.jpg', cv2.IMREAD_GRAYSCALE)
    result_img = boost.face_detection(original_img, scale_step=st)
    cv2.imwrite('Result_img_No_2_%s_plus.png' % boosting_type, result_img)
    original_img = cv2.imread('./test/Face_3.jpg', cv2.IMREAD_GRAYSCALE)
    result_img = boost.face_detection(original_img, scale_step=st)
    cv2.imwrite('Result_img_No_3_%s_plus.png' % boosting_type, result_img)

    #wrong_img = boost.get_hard_negative_patches(original_img, scale_step=st)
    #print(wrong_img.shape)
    #wrong_label = -1 * np.ones((wrong_img[0].shape[0],))
    #data_plus = np.concatenate((data, wrong_img[0]), axis=0)
    #print(data_plus.shape)

    #boost.data = data_plus  # size
    #labels_plus = np.concatenate((labels, wrong_label), axis=0)
    #print(labels_plus.shape)
    #boost.labels = labels_plus  # size
    #act_cache_dir_plus = 'wc_activations_plus.npy' if not flag_subset else 'wc_activations_subset_plus.npy'
    #chosen_wc_cache_dir_plus = 'chosen_wcs_plus.pkl' if not flag_subset else 'chosen_wcs_subset_plus.pkl'
    #boost.calculate_training_activations(act_cache_dir_plus, act_cache_dir_plus)
    #boost.train(chosen_wc_cache_dir_plus)
    #result_img = boost.face_detection(resize_img, scale_step=st)
    #v2.imwrite('Result_img_hard_negative_%s.png' % boosting_type, result_img)

if __name__ == '__main__':
	main()
