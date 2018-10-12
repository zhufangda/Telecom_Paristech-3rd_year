from skimage import color
from skimage import measure
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re
import datetime as dt
import os
import re
import imageio
from dateutil import parser 

def vectorize(path):
    '''
    This function import the image from the path and vectorize it
    
    parameters:
    --------------------------------------------------------
    path :               str, path of the image file
    --------------------------------------------------------
    '''
    map_1 = imageio.imread(path)
    map_2 = color.colorconv.rgb2grey(map_1)
    map_3 = np.flipud(map_2)
    map_contours = measure.find_contours(map_3, 0.8, fully_connected='high')
    return map_contours

            
def print_map(a, map_contours, title):
    '''
    This function print the map in backgroung
    
    parameters:
    --------------------------------------------------------
    a :               the axe to plot
    map_contours :    background
    title :           str, title of the figure
    --------------------------------------------------------
    '''
    a.set_xlim(0, 200)
    a.set_ylim(0, 200)
    a.grid(color=(0.8, 0.8, 0.8), linestyle='-', zorder=1)
    a.scatter(148, 159, marker="$\u2A3B$", s=1000, color='red', 
           label ="dumping site", zorder=10)
    for contour in map_contours:
        a.plot(contour[:, 1], contour[:, 0], 
               linewidth=1, color='#662506', zorder=2)
    a.set_title(title, fontsize=20)
    a.legend()
    return a

def clean_grid(X, drop_bad_value=False):
    '''
    This function clean the grids
    
    parameters:
    --------------------------------------------------------
    X              :   str, value to check
    drop_bad_value :   booleen (False by default)
                       if True, replace the bad_values by -1
                       if False, try to extract number
    --------------------------------------------------------
    '''
    if type(X) is not int:
        a = int(max(re.findall('\d+', X), key=len))
        if not a:
             return -1
        else:
            if drop_bad_value:
                return -1
            else:
                return a
    else:
        if (0 <= X <= 200):
            return X
        else:
             return -1

def clean_time(time) :
    try : 
        parsed_time = parser.parse(time).hour
    except ValueError :
        parsed_time = 0
    return parsed_time
            
def pause(serie_date, nb):
    for i in range(nb):
        serie_date.append(serie_date[len(serie_date)-1])
    return serie_date