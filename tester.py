#/usr/bin/python
'''
This program tests cascade on test data
You need to form 2 files: with positive info and negative info 
(read description of read functions)
    -c: cascade file path
    -t: file with test data
'''
from logging import getLogger, DEBUG, StreamHandler, Formatter
from sys import argv, stdout, exit
from numpy import array
from argparse import ArgumentParser
from RealTimeDepthDetector.Images import load_test_data, to_integral_image
from RealTimeDepthDetector.Utils import load_cascade, ensemble_vote_all



if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        '-c', '--class', 
        help='Classifier file path', 
        action='store', type=str, dest='classifier', required=True
    )
    parser.add_argument(
        '-t', '--test',
        help='Test data file', 
        action='store', type=str, dest='test', required=True
    )
    parser.add_argument(
        '-imgdir', '--imgdir',
        help='Image dir prefix', 
        action='store', type=str, dest='imgdir', required=False
    )
    
    #parse arguments
    args = parser.parse_args(argv[1:])
    imgdir = args.imgdir if args.imgdir else ''
    
    # make logging output    
    mainlog = getLogger()
    mainlog.setLevel(DEBUG)
    ch = StreamHandler(stdout)
    ch.setLevel(DEBUG)
    formatter = Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    mainlog.addHandler(ch)
    
    mainlog.info('Load cascade')
    cascade, winsize = load_cascade(args.classifier)
    
    if len(cascade) == 0:
        mainlog.error('Cant load cascade from file')
        exit()
    
    mainlog.info('Load images')
    positive, negative = load_test_data(args.test, imgdir, winsize)
    
    if not len(positive) and not len(negative):
        mainlog.error('No test data loaded')
        exit()
    
    positive_ii = list(map(to_integral_image, positive))
    negative_ii = list(map(to_integral_image, negative))
    positive_votes = array(ensemble_vote_all(positive_ii, cascade))
    negative_votes = array(ensemble_vote_all(negative_ii, cascade))
    
    posguessed = (positive_votes == 1).sum() / len(positive_votes)
    negguessed = (negative_votes == 0).sum() / len(negative_votes)
    
    mainlog.info("{0:.2f}% of positive guessed right".format(posguessed))
    mainlog.info("{0:.2f}% of negative guessed right".format(negguessed))

