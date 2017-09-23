'''
@author: necrosis
'''
from logging import error
from collections import namedtuple



Rect = namedtuple('Rect', ['x', 'y', 'width', 'height'])
Point = namedtuple('Point', ['x', 'y'])



def area(int_img, rect):
    """
    Calculates depth area
    """
    top_left = rect.x, rect.y
    bottom_right = rect.x + rect.width, rect.y + rect.height
    if top_left == bottom_right:
        return int_img[top_left]
    
    top_right = (bottom_right[0], top_left[1])
    bottom_left = (top_left[0], bottom_right[1])
    try:
        val = int_img[bottom_right] - int_img[top_right] - int_img[bottom_left] + int_img[top_left]
        return val / (rect.width * rect.height)
    except IndexError:
        error("Index error! Check your indexes")
        return 0



class DynamicDepthBasedFeature:
    """
    """
    def __init__(self, img_center, position, size, threshold = 0, weight = 1.):
        """
        Creates a new haar-like feature.
        :param center_depth: Central Depth Value
        :type float:
        :param position: Top-Left porotion of rect
        :type tuple: (int, int)
        :param size: Size of rect
        :type tuple: (int, int)
        :param threshold: Feature threshold
        :type threshold: float
        :param weight: Feature weight
        :type weight: float
        """
        self.center = Rect(*img_center, 1, 1)
        self.rect = Rect(*position, *size)
        self.threshold = threshold
        self.weight = weight
    
    def get_vote(self, int_img):
        """
        Get vote for given integral image array.
        :param int_img: Integral image array
        :type int_img: numpy.ndarray
        :return: Vote for given feature
        :rtype: 1 or 0
        """
        score = self.get_score(int_img)
        return self.weight * (1 if score < self.threshold else -1)
    
    def get_score(self, int_img):
        """
        Get score for given integral image array.
        :param int_img: Integral image array
        :type int_img: numpy.ndarray
        :return: Score for given feature
        :rtype: float
        """
        depth = area(int_img, self.center)
        ar = area(int_img, self.rect)
        score = depth - ar
        
        return score
    
    def make_dict(self):
        """
        Create and return info about feature as dict
        :return: info dict
        :rtype: dict
        """
        return {
            "rect": tuple(self.rect),
            "weight": self.weight,
            "threshold": self.threshold,
            "center": (self.center.x, self.center.y)
        }
    
    @classmethod
    def from_dict(self, data):
        """
        Creates feature object from dict
        :param data: dict with values to create dict
        :type data: dict
        
        :return: new feature
        :rtype: DynamicDepthBasedFeature
        """
        rect = Rect(*data["rect"])
        try:
            return DynamicDepthBasedFeature(
                data["center"],
                (rect.x, rect.y),
                (rect.width, rect.height),
                data["threshold"],
                data["weight"]
            )
        except KeyError:
            error('[FromDict] Wrong json data format')
        except:
            error('[FromDict] Unknown error on creating feature')
        
        return None



__all__ = ["DynamicDepthBasedFeature"]
