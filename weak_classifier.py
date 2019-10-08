from abc import ABC, abstractmethod
import numpy as np
from joblib import Parallel, delayed
import math


class Weak_Classifier(ABC):
    # initialize a harr filter with the positive and negative rects
    # rects are in the form of [x1, y1, x2, y2] 0-index
    def __init__(self, id, plus_rects, minus_rects, num_bins):
        self.id = id
        self.plus_rects = plus_rects
        self.minus_rects = minus_rects
        self.num_bins = num_bins
        self.activations = None

    # take in one integrated image and return the value after applying the image
    # integrated_image is a 2D np array
    # return value is the number BEFORE polarity is applied
    def apply_filter2image(self, integrated_image):
        pos = 0
        for rect in self.plus_rects:
            rect = [int(n) for n in rect]
            pos += integrated_image[rect[3], rect[2]] \
                   + (0 if rect[1] == 0 or rect[0] == 0 else integrated_image[rect[1] - 1, rect[0] - 1]) \
                   - (0 if rect[1] == 0 else integrated_image[rect[1] - 1, rect[2]]) \
                   - (0 if rect[0] == 0 else integrated_image[rect[3], rect[0] - 1])
        neg = 0
        for rect in self.minus_rects:
            rect = [int(n) for n in rect]
            neg += integrated_image[rect[3], rect[2]] \
                   + (0 if rect[1] == 0 or rect[0] == 0 else integrated_image[rect[1] - 1, rect[0] - 1]) \
                   - (0 if rect[1] == 0 else integrated_image[rect[1] - 1, rect[2]]) \
                   - (0 if rect[0] == 0 else integrated_image[rect[3], rect[0] - 1])
        return pos - neg

    # take in a list of integrated images and calculate values for each image
    # integrated images are passed in as a 3-D np-array
    # calculate activations for all images BEFORE polarity is applied
    # only need to be called once
    def apply_filter(self, integrated_images):
        values = []
        for idx in range(integrated_images.shape[0]):
            values.append(self.apply_filter2image(integrated_images[idx, ...]))
        if (self.id + 1) % 100 == 0:
            print('Weak Classifier No. %d has finished applying' % (self.id + 1))
        return values


    # using this function to compute the error of
    # applying this weak classifier to the dataset given current weights
    # return the error and potentially other identifier of this weak classifier
    # detailed implementation is up you and depends
    # your implementation of Boosting_Classifier.train()
    @abstractmethod
    def calc_error(self, weights, labels):
        pass

    @abstractmethod
    def predict_image(self, integrated_image):
        pass


class Ada_Weak_Classifier(Weak_Classifier):
    def __init__(self, id, plus_rects, minus_rects, num_bins):
        super().__init__(id, plus_rects, minus_rects, num_bins)
        self.polarity = None
        self.threshold = None

    def calc_error(self, weights, labels):
        ######################
        ######## TODO ########
        ######################
        w = self.activations
        data_max = np.max(w)
        data_min = np.min(w)

        t = 20
        err_l = []
        err = 1
        step = ((data_max - data_min) / t) * np.ones((labels.shape[0],))
        thresh = data_min * np.ones((labels.shape[0],))
        th = data_min
        tmp = 0
        p = 1
        for i in range(t):
            #l = [np.sign(x - thresh) for x in w]
            l = np.sign(w-thresh)
            #ss = np.zeros((labels.shape[0],))
            ss = np.where(labels!=l,1,0)
            #for n in range(labels.shape[0]):
                #if labels[n] != l[n]:
                    #ss[n] = 1
                    #tmp += weights[i]*1
            tmp = np.sum(np.multiply(weights,ss))
            #tmp = 1 - ((((l - labels) == 0).sum()) / labels.shape[0])
            #tmp /= labels.shape[0]
            if tmp < err:
                err = tmp
                th = thresh[0]
                p = 1
            # err_l.append(tmp)
            thresh += step
        thresh = data_min * np.ones((labels.shape[0],))
        tmp = 0
        for i in range(t):
            #l = [-np.sign(x - thresh) for x in w]
            l = -np.sign(w-thresh)
            ss = np.where(labels!=l,1,0)
            #ss = np.zeros((labels.shape[0],))
            #for n in range(labels.shape[0]):
                #if labels[n] != l[n]:
                    #ss[n] = 1
                    #tmp += weights[i]*1
            tmp = np.sum(np.multiply(weights,ss))
            #tmp = 1 - ((((l - labels) == 0).sum()) / labels.shape[0])
            #tmp /= labels.shape[0]
            if tmp < err:
                err = tmp
                th = thresh[0]
                p = -1
            # err_l.append(tmp)
            thresh += step
        # self.threshold = th
        #print("Find an error")
        # self.polarity = p
        return err, th, p

    def predict_image(self, integrated_image):
        value = self.apply_filter2image(integrated_image)
        return self.polarity * np.sign(value - self.threshold)

    def predict_activations(self):
        v = self.activations
        p = self.polarity * np.ones((v.shape[0],))
        t = self.threshold * np.ones((v.shape[0],))
        return np.multiply(p, np.sign(v-t))


class Real_Weak_Classifier(Weak_Classifier):
    def __init__(self, id, plus_rects, minus_rects, num_bins):
        super().__init__(id, plus_rects, minus_rects, num_bins)
        self.thresholds = None  # this is different from threshold in ada_weak_classifier, think about it
        self.bin_pqs = None
        self.train_assignment = None

    def calc_error(self, weights, labels):
        ######################
        ######## TODO ########
        ######################
        self.thresholds = np.linspace(np.min(self.activations), np.max(self.activations), self.num_bins)
        if not self.train_assignment:
            self.train_assignment = [[] for _ in range(self.num_bins)]
            for i in range(self.activations.shape[0]):
                bin_idx = np.sum(self.thresholds < self.activations[i])
                self.train_assignment[bin_idx].append(i)
        ht = np.zeros(self.activations.shape[0])
        for i in range(self.num_bins):
            ids = self.train_assignment[i]
            self.bin_pqs[0, i] = np.sum(weights[ids] * np.array(labels[ids] == 1, dtype=float)) + 0.0001
            self.bin_pqs[1, i] = np.sum(weights[ids] * np.array(labels[ids] == -1, dtype=float)) + 0.0001
            ht[ids] = 0.5 * np.log(self.bin_pqs[0, i] / self.bin_pqs[1, i])

        #print(self.bin_pqs[0, :])
        #print(self.bin_pqs[1, :])
        #finished assigning qp and thresholds

        return ht

    def predict_image(self, integrated_image):
        value = self.apply_filter2image(integrated_image)
        bin_idx = np.sum(self.thresholds < value) - 1
        return 0.5 * np.log((self.bin_pqs[0, bin_idx] / self.bin_pqs[1, bin_idx])+0.0001)

    def predict_activations(self):
        value = self.activations
        bin_idx = (np.sum(self.thresholds < value,axis=1) - 1)[0,:]
        return 0.5 * np.log(np.divide(self.bin_pqs[0, bin_idx], self.bin_pqs[1, bin_idx]) + 0.001)


def main():
    plus_rects = [(1, 2, 3, 4)]
    minus_rects = [(4, 5, 6, 7)]
    num_bins = 50
    ada_hf = Ada_Weak_Classifier(plus_rects, minus_rects, num_bins)
    real_hf = Real_Weak_Classifier(plus_rects, minus_rects, num_bins)


if __name__ == '__main__':
    main()
