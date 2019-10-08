import os
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
import pickle
from PIL import Image, ImageDraw
import math

import cv2
from weak_classifier import Ada_Weak_Classifier, Real_Weak_Classifier
from im_process import image2patches, nms, normalize


class Boosting_Classifier:
    def __init__(self, haar_filters, data, labels, num_chosen_wc, num_bins, visualizer, num_cores, style):
        self.filters = haar_filters
        self.data = data
        self.labels = labels
        self.num_chosen_wc = num_chosen_wc
        self.num_bins = num_bins
        self.visualizer = visualizer
        self.num_cores = num_cores
        self.style = style
        self.chosen_wcs = None
        if style == 'Ada':
            self.weak_classifiers = [Ada_Weak_Classifier(i, filt[0], filt[1], self.num_bins) \
                                     for i, filt in enumerate(self.filters)]
        elif style == 'Real':
            self.weak_classifiers = [Real_Weak_Classifier(i, filt[0], filt[1], self.num_bins) \
                                     for i, filt in enumerate(self.filters)]

    def calculate_training_activations(self, save_dir=None, load_dir=None):
        print('Calcuate activations for %d weak classifiers, using %d imags.' % (
        len(self.weak_classifiers), self.data.shape[0]))
        if load_dir is not None and os.path.exists(load_dir):
            print('[Find cached activations, %s loading...]' % load_dir)
            wc_activations = np.load(load_dir)
        else:
            if self.num_cores == 1:
                wc_activations = [wc.apply_filter(self.data) for wc in self.weak_classifiers]
            else:
                wc_activations = Parallel(n_jobs=self.num_cores, verbose=10)(
                    delayed(wc.apply_filter)(self.data) for wc in self.weak_classifiers)
            wc_activations = np.array(wc_activations)
            if save_dir is not None:
                print('Writing results to disk...')
                np.save(save_dir, wc_activations)
                print('[Saved calculated activations to %s]' % save_dir)
        for wc in self.weak_classifiers:
            wc.activations = wc_activations[wc.id, :]
        return wc_activations

    # select weak classifiers to form a strong classifier
    # after training, by calling self.sc_function(), a prediction can be made
    # self.chosen_wcs should be assigned a value after self.train() finishes
    # call Weak_Classifier.calc_error() in this function
    # cache training results to self.visualizer for visualization
    #
    #
    # detailed implementation is up to you
    # consider caching partial results and using parallel computing
    def train(self, save_dir=None):
        ######################
        ######## TODO ########
        ######################
        print("Training start")
        self.visualizer.labels = self.labels
        if self.style == 'Ada':
            self.chosen_wcs = []
            D = np.ones((self.data.shape[0],)) / self.data.shape[0]
            acc = []
            p_score = np.zeros((self.labels.shape[0],))
            c_id = []
            weak_acc = np.zeros((1000,))
            for t in range(self.num_chosen_wc):
                chose_err = 1
                sum_weak = 0
                chose_id = 0
                count = 0
                if t % 20 == 0:
                    print("Step "+str(t)+"Completed!")
                for weak in self.weak_classifiers:
                    if weak.id not in c_id:
                        err, th, p = weak.calc_error(D, self.labels)
                        weak.polarity = p
                        weak.threshold = th
                        #print("One Done")
                        if count < 1000:
                            weak_acc[count] = err
                        count += 1
                        if ((err < chose_err) and (weak.id not in c_id)):
                            chose_err = err
                            chose_id = weak.id
                            c_id.append(chose_id)
                if (t == 0 or t == 10 or t == 50):
                    self.visualizer.weak_classifier_accuracies[t] = np.array(weak_acc)
                if (chose_err < 0.5):
                    print("Choose Id: " + str(chose_id))
                    print("Selected Err: " + str(chose_err))
                    a = 0.5 * (math.log((1 - chose_err) / chose_err))
                    print("Selected alpha: " + str(a))
                    #alpha[t] = a
                    chose = self.weak_classifiers[chose_id]
                    #assert chose.id == chose_id
                    self.chosen_wcs.append((a, chose))
                    #Zt = 0
                    Zt = np.sum(np.multiply(D, np.exp(a*np.multiply(-self.labels, chose.predict_activations()))))
                    #for i in range(self.labels.shape[0]):
                        #Zt += D[i] * math.exp(-self.labels[i] * a * chose.predict_image(self.data[i]))  # h(xi)???
                    #print("Zt: " + str(Zt))
                    #for i in range(self.labels.shape[0]):
                        #D[i] = (1 / Zt) * D[i] * math.exp(-self.labels[i] * a * chose.predict_image(self.data[i]))
                    D = (1 / Zt) * np.multiply(D, np.exp(a*np.multiply(-self.labels, chose.predict_activations())))
                    #print("Sum of D = "+str(np.sum(D)))
                    #print(D)
                else:
                    print("Cannot find 0.5 Error")
                    return

                #for i in range(self.labels.shape[0]):
                    #p_score[i] = self.sc_function(self.data[i])
                p_score = self.sc_function_s()
                if t == 10 or t == 50:
                    self.visualizer.strong_classifier_scores[t] = np.array(p_score)
                print(p_score.shape)
                l = np.sign(p_score)
                #l = [np.sign(x - thresh) for x in p_score]
                #l = np.array(l)
                tmp = np.sum(np.where(self.labels==l,1,0)) / self.labels.shape[0]
                acc.append(tmp)
            print(acc)
            self.visualizer.draw_strong_acc(acc)
            self.visualizer.weak_classifier_accuracies[100] = np.array(weak_acc)
            self.visualizer.strong_classifier_scores[100] = np.array(p_score)
            self.visualizer.draw_rocs()
            self.visualizer.draw_histograms()
            self.visualizer.draw_wc_accuracies()
            #weak_acc????
        else:
            load = 'chosen_wcs.pkl'
            chosen_ada_wcs = pickle.load(open(load, 'rb'))
            D = np.ones((self.data.shape[0],)) / self.data.shape[0]
            p_score = np.zeros((self.labels.shape[0],))
            #step = [10,50,100]
            step = [10]
            self.chosen_wcs = []
            acc = []
            #for t in step:
            for t in range(self.num_chosen_wc):
                #if t % 20 == 0:
                    #print("Step " + str(t) + "Completed!")
                chose_id = chosen_ada_wcs[t][1].id
                #print("Choose Id: " + str(chose_id))
                chose = self.weak_classifiers[chose_id]
                #chose.activations = chosen_ada_wcs[t][1].activations
                ht = chose.calc_error(D, self.labels)
                self.chosen_wcs.append(chose)
                Zt = np.sum(D*np.exp(-self.labels*ht))
                D = D * np.exp(-self.labels*ht) /Zt
                #Zt = 0
                #Zt = np.sum(np.multiply(D, np.exp(np.multiply(-self.labels, chose.predict_activations()))))
                #for i in range(self.labels.shape[0]):
                    #Zt += D[i] * np.exp(-self.labels[i]*chose.predict_image(self.data[i]))
                #print("Zt: " + str(Zt))
                #for i in range(self.labels.shape[0]):
                    #D[i] = (1/Zt)*D[i]*np.exp(-self.labels[i]*chose.predict_image(self.data[i]))
                #print("Sum of D = " + str(np.sum(D)))
                #D = (1/Zt) * np.multiply(D, np.exp(np.multiply(-self.labels, predict_activations())))
                #for i in range(self.labels.shape[0]):
                    #p_score[i] = np.sum([np.array([wc.predict_image(self.data[i]) for wc in self.chosen_wcs])])
                #p_score = self.sc_function_s()
                #a = self.chosen_wcs[0].predict_image(self.data[0])
                #print(a)
                if t == 10 or t == 50:
                    #s = 0
                    for i in range(self.labels.shape[0]):
                        #s += np
                        p_score[i] = np.sum([np.array([wc.predict_image(self.data[i]) for wc in self.chosen_wcs])])
                    self.visualizer.strong_classifier_scores[t] = np.array(p_score)
                print("Step No. "+str(t)+" OK")
            #print(acc)
            for i in range(self.labels.shape[0]):
                p_score[i] = np.sum([np.array([wc.predict_image(self.data[i]) for wc in self.chosen_wcs])])
            self.visualizer.strong_classifier_scores[100] = np.array(p_score)
            self.visualizer.draw_histograms()
            self.visualizer.draw_rocs()
            self.visualizer.draw_strong_acc(acc)

        if save_dir is not None:
            pickle.dump(self.chosen_wcs, open(save_dir, 'wb'))

    def sc_function(self, image):
        return np.sum([np.array([alpha * wc.predict_image(image) for alpha, wc in self.chosen_wcs])])

    def sc_function_s(self):
        return np.sum([np.array([alpha * wc.predict_activations() for alpha, wc in self.chosen_wcs])],axis=1)[0, :]

    def load_trained_wcs(self, save_dir):
        self.chosen_wcs = pickle.load(open(save_dir, 'rb'))

    def face_detection(self, img, scale_step=20):

        # this training accuracy should be the same as your training process,
        ##################################################################################
        train_predicts = []
        for idx in range(self.data.shape[0]):
            train_predicts.append(self.sc_function(self.data[idx, ...]))
        print('Check training accuracy is: ', np.mean(np.sign(train_predicts) == self.labels))
        ##################################################################################

        scales = 1 / np.linspace(1, 8, scale_step)
        patches, patch_xyxy = image2patches(scales, img)
        print('Face Detection in Progress ..., total %d patches' % patches.shape[0])
        predicts = [self.sc_function(patch) for patch in tqdm(patches)]
        print(np.mean(np.array(predicts) > 0), np.sum(np.array(predicts) > 0))
        pos_predicts_xyxy = np.array([patch_xyxy[idx] + [score] for idx, score in enumerate(predicts) if score > 0])
        if pos_predicts_xyxy.shape[0] == 0:
            return
        print(pos_predicts_xyxy.shape)
        xyxy_after_nms = nms(pos_predicts_xyxy, 0.01)

        print('after nms:', xyxy_after_nms.shape[0])
        for idx in range(xyxy_after_nms.shape[0]):
            pred = xyxy_after_nms[idx, :]
            cv2.rectangle(img, (int(pred[0]), int(pred[1])), (int(pred[2]), int(pred[3])), (0, 255, 0),
                          2)  # gree rectangular with line width 3

        return img

    def get_hard_negative_patches(self, img, scale_step=10):
        scales = 1 / np.linspace(1, 8, scale_step)
        patches, patch_xyxy = image2patches(scales, img)
        print('Get Hard Negative in Progress ..., total %d patches' % patches.shape[0])
        predicts = [self.sc_function(patch) for patch in tqdm(patches)]
        predicts = np.array(predicts)

        wrong_patches = patches[np.where(predicts > 0), ...]

        return wrong_patches

    def visualize(self):
        self.visualizer.labels = self.labels
        self.visualizer.draw_histograms()
        self.visualizer.draw_rocs()
        self.visualizer.draw_wc_accuracies()

    def draw_filter(self,f,num,p):
        if p == 1:
            color_1 = 'black'
            color_2 = 'green'
        else:
            color_1 = 'green'
            color_2 = 'black'
        img = Image.new('RGB',(16,16),(255,255,255))
        dr = ImageDraw.Draw(img)
        dr.rectangle(list(f[0][0]),fill=color_1)
        dr.rectangle(list(f[1][0]),fill=color_2)
        img.save("Filter_No."+str(num)+".png")
