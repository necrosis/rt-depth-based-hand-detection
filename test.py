'''
@author: golenkovv
'''
#from logging import info
#from DepthHandDetection.Images import load_depth_images, to_integral_image, load_pos_from_file
from logging import info, getLogger, DEBUG, StreamHandler, Formatter
from sys import stdout
from itertools import combinations_with_replacement as comb
from RealTimeDepthDetector.Train import training
from RealTimeDepthDetector.Images import load_pos_from_file, load_neg_from_file2, to_integral_image
from RealTimeDepthDetector.Utils import ensemble_vote_all, save_cascade

if __name__ == '__main__':
    
    root = getLogger()
    root.setLevel(DEBUG)
    ch = StreamHandler(stdout)
    ch.setLevel(DEBUG)
    formatter = Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    root.addHandler(ch)
    
    num_classifiers = 48
    
    positive, alpha, r = load_pos_from_file('d:\\temp\\data_kids_3\\rezult\\subpos.txt', 'd:\\temp\\data_kids_3')
    negative = load_neg_from_file2('d:\\temp\\data_kids_3\\rezult\\subneg.txt', '', r)
    
    positive_ii = list(map(to_integral_image, positive))
    negative_ii = list(map(to_integral_image, negative))
    
    ns = list(comb((2, 3, 4, 6, 8), 2))
    feature_sizes = ((r, r), )
    classifiers = training(positive_ii, negative_ii, num_classifiers, ns)
    
    info(len(classifiers))
    
    positive.clear()
    negative.clear()
    positive_ii.clear()
    negative_ii.clear()
    
    positive, alpha, rr = load_pos_from_file('d:\\temp\\data_kids_3\\rezult\\test_subpos.txt', 'd:\\temp\\data_kids_3')
    negative = load_neg_from_file2('d:\\temp\\data_kids_3\\rezult\\test_subneg.txt', '', rr)
    
    positive_ii = list(map(to_integral_image, positive))
    negative_ii = list(map(to_integral_image, negative))
    
    info("Pos = {0}".format(ensemble_vote_all(positive_ii, classifiers)))
    info("Neg = {0}".format(ensemble_vote_all(negative_ii, classifiers)))
    
    
    save_cascade(classifiers, (r, r), "D:\\classifiers.json")
