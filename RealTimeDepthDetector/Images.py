'''
@author: necrosis
'''
from logging import error
from itertools import product
from os.path import exists, join
from os import listdir
import numpy as np
from RealTimeDepthDetector.Utils import get_representative_depth



def to_integral_image(img):
    """
    Calculates the integral image based on this instance's original image data.
    :param img: Image source data
    :type img: numpy.ndarray
    :return Integral image for given image
    :rtype: numpy.ndarray
    """
    return img.cumsum(axis=0).cumsum(axis=1)



def load_raw(img_path):
    """
    Loading raw depth image from my format file
    [header | width | height | data(width*height)]
      int16   int32   int32       of int16
    I've saved raw data from kinect to this format.
    If you know better formats, replace this function
    :param img_path: path to raw depth image file
    :type img_path: string
    
    :return: np.array with raw depth (one row)
    :rtype: np.array
    """
    if not exists(img_path):
        error('[LoadRaw] No image {0}'.format(img_path))
        return None
    
    g = open(img_path, 'rb')
    code = np.fromfile(g, dtype=np.int16, count=1)
    
    if code != 0xd:
        error('[LoadRaw] Wrong raw format {0}'.format(img_path))
        return None
    
    width = int(np.fromfile(g, dtype=np.int32, count=1))
    height = int(np.fromfile(g, dtype=np.int32, count=1))
    count = width*height
    rawData = np.array(
        np.fromfile(g, dtype=np.int16, count=count), dtype=np.float64
    )
    
    g.close()
    return rawData.reshape((height, width))



def load_pos_from_file(filepath, imagedir):
    """
    Loads positive raw depth images from the list on file
    Format of list in file:
    PathToFile 1 x, y, w, h
    It's a copy of opencv pos file, but only for 1 region
    in image. Sorry. It's because of ContourFilterTool
    AND
    Also - calculates optimal image region through 
    second polinomial model
    :param filepath: path to file with list of images
    :type filepath: string
    :param imagedir: image dir, which will be joined to each filepath if file
    :type imagedir: string
    
    :return: list of image regions, alpha vec and optimal region size
    :rtype: list(np.array, ...), np.array, float
    """
    if not exists(filepath):
        error('No image {0}'.format(filepath))
        return None
    
    loaded_data = list()
    P_data = list()
    Y_data = list()
    
    with open(filepath, 'r') as pos_file:
        for pos_line in pos_file.readlines():
            pos_path, _, x, y, w, h = pos_line.split(' ', maxsplit=6)
            x, y, w, h = map(int, (x, y, w, h))
            
            load_path = join(imagedir, pos_path)
            raw_image = load_raw(load_path)
            if raw_image is not None:
                pos_part = raw_image[y:y+h, x:x+w]
                #images.append(pos_part)
                
                depth = get_representative_depth(pos_part)
                r = min(w, h)
                
                P_data.append((1, depth, depth**2))
                Y_data.append(r)
                loaded_data.append((load_path, x, y, depth))
    
    P = np.array(P_data)
    Y = np.array(Y_data)
    Pt = np.transpose(P)
    
    temp = np.dot(Pt, P)
    temp = np.dot(np.linalg.inv(temp), Pt)
    alpha = np.dot(temp, Y)
    
    average_r = 0.0
    for load_path, _, _, depth in loaded_data:
        average_r += round(np.dot(alpha, (1, depth, depth**2)))
    
    average_r = int(round(average_r / len(loaded_data)))
    images = list()        
    for load_path, x, y, _ in loaded_data:
        raw_image = load_raw(load_path)
        
        if raw_image is not None:
            images.append(raw_image[y:y+average_r, x:x+average_r])
    
    return images, alpha, average_r
    for load_path, _, _, depth in loaded_data:
        average_r += round(np.dot(alpha, (1, depth, depth**2)))
    
    average_r = int(round(average_r / len(loaded_data)))
    half_r1 = int(r / 2)
    half_r2 = average_r - half_r1
    images = list()        
    for load_path, x, y, _ in loaded_data:
        raw_image = load_raw(load_path)
        
        if raw_image is not None:
            images.append(raw_image[y-half_r1:y+half_r2, x-half_r1:x+half_r2])
    
    return images, alpha, average_r



def load_neg_from_file(negpath, imagedir, sp=(-1, -1)):
    """
    Loads negative raw depth images from the list in file
    Every negative image will be divided on segments of
    sp size, with step sp/2. Beware of adding too many
    files in this list - it takes a lot of memorys
    :param negpath: path to file with neg list
    :type negpath: string
    :param imagedir: image dir, which will be joined to each filepath if file
    :type imagedir: string
    :param sp: size of segment
    :type sp: (w, h)
    
    :return: list of segments
    :rtype: list(np.array)
    """
    if not exists(negpath):
        error('No image {0}'.format(negpath))
        return None
    
    images = list()
    
    with open(negpath, 'r') as pos_file:
        for pos_line in pos_file.readlines():
            img_path = pos_line.strip()
            raw_image = load_raw(join(imagedir, img_path))
            
            if raw_image is None:
                continue
            
            if sp[0] < 0 or sp[1] < 0:
                images.append(raw_image)
            else:
                h, w = raw_image.shape
                for x, y in product(range(0, w-sp[0]), range(0, h-sp[1])):
                    images.append(raw_image[y:y+sp[1], x:x+sp[0]])
    
    return images



def load_neg_from_file2(negpath, imagedir, r):
    """
    Analogue of load_pos_from_file, but for negative
    and without calculating region size
    :param negpath: path to file with list of images
    :type negpath: string
    :param imagedir: image dir, which will be joined to each filepath if file
    :type imagedir: string
    :param r: region size
    :type r: tuple(w, h)
    
    :return: list of image regions
    :rtype: list(np.array, ...)
    """
    if not exists(negpath):
        error('No image {0}'.format(negpath))
        return None
    
    images = list()
    with open(negpath, 'r') as neg_file:
        for neg_line in neg_file.readlines():
            neg_path, _, x, y, _, _ = neg_line.split(' ', maxsplit=6)
            x, y = map(int, (x, y))
            
            raw_image = load_raw(join(imagedir, neg_path))
            
            if raw_image is None:
                continue
            
            neg_part = raw_image[y:y+r, x:x+r]
            images.append(neg_part)
    
    return images



def load_test_data(fpath, imagedir, r):
    """
    Loads test data 
    :param fpath: path to file
    :type fpath: str
    :param imagedir: prefix to all filepathes
    :type imagedir: str
    :param r: region size
    :type r: (w, h)
    """
    if not exists(fpath):
        error('No image {0}'.format(fpath))
        return None
    
    pos_images = list()
    neg_images = list()
    with open(fpath, 'r') as neg_file:
        for neg_line in neg_file.readlines():
            neg_path, S, x, y, w, h = neg_line.split(' ', maxsplit=6)
            x, y, S, w, h = map(int, (x, y, S, w, h))
            x, y = int(x+w/2), int(y+h/2)
            hr1 = int(r[0]/2), int(r[1]/2)
            hr2 = r[0]-hr1[0], r[1]-hr1[1]
            
            raw_image = load_raw(join(imagedir, neg_path))
            
            if raw_image is None:
                error('')
                continue
            
            part = raw_image[y-hr1[0]:y+hr2[0], x-hr1[1]:x+hr2[1]]
            if S == 0:
                neg_images.append(part)
            else:
                pos_images.append(part)
    
    return pos_images, neg_images


def load_depth_images(path):
    """
    Load all raw images in dir
    :param path: path to dir
    :type path: string
    
    :return: list of images
    :rtype: list(np.array, ...)
    """
    images = []
    for _file in listdir(path):
        if _file.endswith('.raw'):
            img_arr = load_raw(join(path, _file))
            img_arr /= img_arr.max()
            images.append(img_arr)
    return images



__all__ = [
    "load_raw", 
    "to_integral_image", 
    "load_pos_from_file",
    "load_neg_from_file",
    "load_neg_from_file2",
    "load_depth_images"
]