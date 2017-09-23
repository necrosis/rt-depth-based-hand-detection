'''
@author: necrosis
'''
from logging import getLogger
from functools import partial
from itertools import product
from multiprocessing import Pool
import numpy as np
from progressbar import ProgressBar
from RealTimeDepthDetector.Features import DynamicDepthBasedFeature
from RealTimeDepthDetector.Utils import goldenSearch



def training(
        positive_iis, 
        negative_iis, 
        num_classifiers, 
        ns, 
        Ddesired=0.9, 
        Edesired = 0.05
    ):
    """
    Selects a set of classifiers. Iteratively takes the best classifiers based
    on a weighted error.
    :param positive_iis: List of positive integral image examples
    :type positive_iis: list[numpy.ndarray]
    :param negative_iis: List of negative integral image examples
    :type negative_iis: list[numpy.ndarray]
    :param num_classifiers: Number of classifiers to select, -1 will use all
    classifiers
    :type num_classifiers: int
    :param ns: List of features widths/heights
    :type ns: [(int,int), (int, int) , ...]
    :param Ddesired: desired detection rate
    :type Ddesired: float
    :param Edesired: desired error rate
    :type Edesired: float

    :return: List of selected features
    :rtype: list[DynamicDepthBasedFeature]
    """
    logger = getLogger('Train')
    
    num_pos = len(positive_iis)
    num_neg = len(negative_iis)
    num_imgs = num_pos + num_neg
    img_height, img_width = positive_iis[0].shape
    
    pos_weights = np.ones(num_pos) * 1. / (2 * num_pos)
    neg_weights = np.ones(num_neg) * 1. / (2 * num_neg)
    weights = np.hstack((pos_weights, neg_weights))
    labels = np.hstack((np.ones(num_pos), -np.ones(num_neg)))
    
    images = positive_iis + negative_iis
    
    logger.info('Stage 1: Creating features')
    features = create_dynamic_features(
        (img_width, img_height),  
        ns,
        0
    )
    num_features = len(features)
    num_classifiers = num_features if num_classifiers == -1 else num_classifiers
    
    logger.info('Generated features: {0}'.format(num_features))
    logger.info('Stage 2: Calculating thresholds')
    
    bar = ProgressBar()
    for i in bar(range(num_features)):
        feature = features[i]
        scores = np.fromiter(
            (feature.get_score(img) for img in images), 
            np.float
        )
        feature.threshold = get_appropriet_threshold(
            scores[:num_pos], 
            scores[num_pos:], 
            Ddesired, 
            Edesired
        )
    
    logger.info('Stage 3: Calculating scores for all images')
    votes = np.zeros((num_imgs, num_features))
    bar = ProgressBar()
    # Use as many workers as there are CPUs
    pool = Pool(processes=None)
    for i in bar(range(num_imgs)):
        features_matrix = np.array( 
            list(
                pool.map(
                #map(   
                    partial(_get_feature_values, image=images[i]), 
                    features
        )))
        # Convert matrix to vector
        votes[i, :] = features_matrix
    
    logger.info('Stage 4: Selecting classifiers')
    classifiers = list()
    feature_indexes = list(range(num_features))
    bar = ProgressBar()
    for _ in bar(range(num_classifiers)):

        classification_errors = np.zeros(len(feature_indexes))
        # normalize weights
        weights *= 1. / np.sum(weights)

        # select best classifier based on the weighted error
        for f in range(len(feature_indexes)):
            f_idx = feature_indexes[f]
            # classifier error is the sum of image weights where the classifier
            # is right
            error = sum( map(
                lambda img_idx: weights[img_idx] if labels[img_idx] != votes[img_idx, f_idx] else 0, 
                range(num_imgs)
            ))
            classification_errors[f] = error

        # get best feature, i.e. with smallest error
        min_error_idx = np.argmin(classification_errors)
        best_error = classification_errors[min_error_idx]
        best_feature_idx = feature_indexes[min_error_idx]

        # set feature weight
        best_feature = features[best_feature_idx]
        eb_eq = (1-best_error)/best_error
        feature_weight = 0.5 * np.log((1 - best_error) / best_error)
        best_feature.weight = feature_weight

        classifiers.append(best_feature)
        
        weights = np.array(
            list(
                map(
                    lambda img_idx: weights[img_idx] * np.sqrt(eb_eq if (labels[img_idx] != votes[img_idx, best_feature_idx]) else 1/eb_eq),
                    range(num_imgs)
                )
            )
        )
        
        feature_indexes.remove(best_feature_idx)
        
    logger.info(
        'Training finished. {0} week classifiers formed'.format(len(classifiers))
    )
    return classifiers



def get_appropriet_threshold(positive, negative, D, E):
    """
    This function founds a threshold value, which make a feature
    to find desired amount of positive examples, while minimizing
    detection error
    :param positive: feature scores for positive images
    :type positive: list(float, ...)
    :param negative: feature scores for negative images
    :type negative: list(float, ...)
    :param D: Desired detection rate
    :type D: float
    :param E: Desired error rate
    :type E: float
    
    :return: threshold
    :rtype: float
    """
    threshold = positive.max()
    threshold_border = positive.min()
    searchArea =  abs(threshold - threshold_border)
    es = len(negative)
    pos = len(positive)
    
    # define the area for desired positive rate
    dleft = goldenSearch(
        lambda t: abs((positive <= t).sum() / pos - D),
        threshold_border,
        threshold_border + searchArea / 100,
        threshold,
        0.08
    )
    
    # in this are search for minimum error
    searchArea = abs(threshold - dleft)
    threshold = goldenSearch(
        lambda t: abs((negative <= t).sum() / es - E), 
        dleft, 
        dleft + searchArea / 100, 
        threshold, 
        0.08
    )
    
    return threshold



def create_dynamic_features(img_size, ns, threshold):
    """
    Creates DynamicDepthBasedFeature features
    :param img_size: size of hand region
    :type img_size: tuple(w, h)
    :param ns: list of inner block's sizes
    :type ns: [(int,int), (int, int) , ...]
    :param threshold: base threshold
    :type threshold: float
    
    :return: list of features
    :rtype: list(DynamicDepthBasedFeature)
    """
    img_width, img_height = img_size
    center = int(img_width / 2), int(img_height / 2)
    
    features = []
    for nx, ny in ns:
        step = int(img_width / (nx << 1)), int(img_height / (ny << 1))
        block = int(img_width / nx), int(img_height / ny)
        end = (nx << 1) - 1, (ny << 1) - 1
        
        for x, y in product(range(end[0]), range(end[1])):
            position = (x*step[0], y*step[1])
            feature = DynamicDepthBasedFeature(center, position, block, threshold)
            features.append(feature)
    
    return features



def _get_feature_values(feature, image):
    return feature.get_vote(image)



def _get_feature_scores(feature, image):
    return feature.get_score(image)



__all__ = ["training"]