'''
@author: necrosis
'''
from math import sqrt
from functools import partial
from logging import info, error
import numpy as np
from json import dumps, loads
from RealTimeDepthDetector.Features import DynamicDepthBasedFeature 



phi = (1 + sqrt(5))/2
resphi = 2 - phi



def ensemble_vote(int_img, classifiers):
    """
    Classifies given integral image (numpy array) using given classifiers, i.e.
    if the sum of all classifier votes is greater 0, image is classified
    positively (1) else negatively (0). The threshold is 0, because votes can be
    +1 or -1.
    :param int_img: Integral image to be classified
    :type int_img: numpy.ndarray
    :param classifiers: List of classifiers
    :type classifiers: list[DynamicDepthBasedFeature]
    
    :return: 1 iff sum of classifier votes is greater 0, else 0
    :rtype: int
    """
    return 1 if sum([c.get_vote(int_img) for c in classifiers]) >= 0 else 0



def ensemble_vote_all(int_imgs, classifiers):
    """
    Classifies given list of integral images (numpy arrays) using classifiers,
    i.e. if the sum of all classifier votes is greater 0, an image is classified
    positively (1) else negatively (0). The threshold is 0, because votes can be
    +1 or -1.
    :param int_imgs: List of integral images to be classified
    :type int_imgs: list[numpy.ndarray]
    :param classifiers: List of classifiers
    :type classifiers: list[DynamicDepthBasedFeature]
    
    :return: List of assigned labels, 1 if image was classified positively, else
    0
    :rtype: list[int]
    """
    vote_partial = partial(ensemble_vote, classifiers=classifiers)
    return list(map(vote_partial, int_imgs))



def goldenSearch(f, a, c, b, absolutePrecision):
    """
    Golden search optimization algorythm
    :param f: function
    :type f: callable object
    :param a: left border
    :type a: float
    :param c: center pointer pushed slightly left towards a
    :type c: float
    :param b: right border
    :type b: float
    :param absolutePrecision: search precision
    :type absolutePrecision: float
    
    :return: representative depth
    :rtype: float
    """
    if abs(a-b) < absolutePrecision:
        return (a+b) / 2
    
    d = c + resphi*(b - c)
    
    if f(d) < f(c):
        return goldenSearch(f, c, d, b, absolutePrecision)
    else:
        return goldenSearch(f, d, c, a, absolutePrecision)



def get_representative_depth(hand_region):
    """
    As a representative depth i've take an average depth of a hand
    I suppose, that hand takes most part of region, and I get
    the most wide spread depth in region through histograms
    :param hand_region: array with depth region with hand
    :type hand_region: np.array
    
    :return: representative depth
    :rtype: float
    """
    rangev = (hand_region.min(), hand_region.max())
    h = np.histogram(hand_region, bins="doane", range=rangev, normed=True)
    
    val, rang = h
    mx = max(val)
    try:
        where = np.where(val==mx)
        if isinstance(where, tuple):
            where = where[0]
        itemindex = int(where[0])
    except TypeError:
        error('Error while calculating representative depth')
        return None
            
    i = max(itemindex-1, 0)
    
    while i > 0 and val[i] > 0.0001:
        i -= 1
        
    return int(rang[i])



def save_cascade(cascade, winSize, filepath):
    """
    :param cascade: list of cascade features
    :type cascade: list(DynamicDepthBasedFeature)
    :param winSize: hand region size 
    :type winSize: tuple(width, height)
    :param filepath: path to json file
    :type filepath: string
    """
    with open(filepath, 'w') as f:
        data = dict()
        data["searchWinSize"] = winSize
        data["features"] = list()
        
        for feature in cascade:
            feature_data = feature.make_dict()
            data["features"].append(feature_data)
        
        f.write(dumps(data, indent=4, sort_keys=True))
    
    info("Cascade saved {0}".format(filepath))



def load_cascade(filepath):
    """
    :param filepath: path to json file
    :type filepath: string
    
    :return: List of features and hand region size
    :rtype: tuple(list(DynamicDepthBasedFeature), (width, height))
    """
    try:
        with open(filepath, 'r') as f:
            features = list()
            dict_data = loads(f.read())
            
            winSize = tuple(dict_data["searchWinSize"])
            feature_list = dict_data["features"]
            
            for feature_data in feature_list:
                feature = DynamicDepthBasedFeature.from_dict(feature_data)
                if not feature is None:
                    features.append(feature)
            
            info('{0} features Loaded'.format(len(features)))
        
            return features, winSize
    except FileNotFoundError:
        error('File not found {0}'.format(filepath))
    
    return list(), (0, 0)



__all__ = [
    "ensemble_vote",
    "ensemble_vote_all",
    "goldenSearch",
    "get_representative_depth",
    "save_cascade",
    "load_cascade"
]