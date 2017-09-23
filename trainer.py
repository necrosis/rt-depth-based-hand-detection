#/usr/bin/python
'''
This program trains classifiers into cascade
You need to form 2 files: with positive info and negative info 
(read description of read functions)
    -p: positive file
    -n: negative file
Also you should specify:
    -o: output file
    -num: amount of weak classifiers
    -d: detect rate (0.9)
    -e: error rate (0.05)
    -nx: list on parts (Nx, Ny) - will be combined
'''
from logging import info, getLogger, DEBUG, StreamHandler, Formatter
from sys import argv, stdout, exit
from itertools import combinations_with_replacement as comb
from argparse import ArgumentParser
from RealTimeDepthDetector.Train import training
from RealTimeDepthDetector.Images import load_pos_from_file, load_neg_from_file2, to_integral_image
from RealTimeDepthDetector.Utils import save_cascade



if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        '-p', '--pos', 
        help='Positive files list', 
        action='store', type=str, dest='pos', required=True
    )
    parser.add_argument(
        '-n', '--neg',
        help='Negative files list', 
        action='store', type=str, dest='neg', required=True
    )
    parser.add_argument(
        '-num', '--num',
        help='Number of classifiers', 
        action='store', type=int, dest='num', required=True
    )
    parser.add_argument(
        '-o', '--output',
        help='Output file to save classifiers', 
        action='store', type=str, dest='output', required=True 
    )
    parser.add_argument(
        '-d', '--detect',
        help='Detection Rate', 
        action='store', type=float, dest='d', required=False
    )
    parser.add_argument(
        '-e', '--error',
        help='Error Rate', 
        action='store', type=float, dest='e', required=False
    )
    parser.add_argument(
        '-x', '--nx',
        nargs='+', help='Error Rate', 
        action='store', type=int, dest='nx', required=True
    )
    parser.add_argument(
        '-imgdir', '--imgdir',
        help='Image dir prefix', 
        action='store', type=str, dest='imgdir', required=False
    )
    
    #parse arguments
    args = parser.parse_args(argv[1:])
    Edesired = args.e if args.e else 0.05
    Ddesired = args.d if args.d else 0.9
    imgdir = args.imgdir if args.imgdir else ''
    
    # make logging output    
    mainlog = getLogger()
    mainlog.setLevel(DEBUG)
    ch = StreamHandler(stdout)
    ch.setLevel(DEBUG)
    formatter = Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    mainlog.addHandler(ch)
    
    #start
    positive, alpha, r = load_pos_from_file(args.pos, imgdir)
    negative = load_neg_from_file2(args.neg, imgdir, r)
    
    if not len(positive) or not len(negative):
        mainlog.error("Training images are not loaded!")
        exit()
    
    positive_ii = list(map(to_integral_image, positive))
    negative_ii = list(map(to_integral_image, negative))
    ns = list(comb(args.nx, 2))
    
    classifiers = training(positive_ii, negative_ii, args.num, ns)
    
    info('Cascade formed from {0} weak classifiers'.format(len(classifiers)))
    save_cascade(classifiers, (r, r), args.output)
