import glob
import numpy as np
from io import StringIO
import xml.etree.ElementTree as ET
import os
from tqdm import tqdm
from skimage import transform
from sklearn.neural_network import MLPClassifier
import csv
from sklearn.externals import joblib


def _normalize_coordinates(coords, maintain_ar = False):
    min_x = np.min(coords[:,0])
    min_y = np.min(coords[:,1])
    
    normalized = coords.copy()
    normalized[:,0] = normalized[:,0] - min_x
    normalized[:,1] = normalized[:,1] - min_y
    
    return normalized

def _coords_to_image(coords, clean=True):
    normalized = _normalize_coordinates(coords)
    
    M = np.max(normalized[:,0]) + 1
    N = np.max(normalized[:,1]) + 1
    
    image = np.zeros((M,N))
        
    for coord in normalized:
        image[coord[0],coord[1]] = 1

    return image.transpose((1,0))
    
def _stroke_to_arr(stroke):
    stroke = stroke.replace(',', '\n')
    stroke_IO = StringIO(stroke)
    stroke_arr = np.loadtxt(stroke_IO)
    return stroke_arr

def _parse_meta(inkml_file):
    tree = ET.parse(inkml_file)
    root = tree.getroot()
    
    annotation = root.find("{http://www.w3.org/2003/InkML}annotation")
    return annotation.text
    
def _parse_traces(inkml_file):
    tree = ET.parse(inkml_file)
    root = tree.getroot()
    traces = root.findall("{http://www.w3.org/2003/InkML}trace")
    d_traces = []
    for trace in traces:
        d_trace = _stroke_to_arr(trace.text)
        if(len(d_trace.shape)==1):
            d_trace = d_trace.reshape(1,d_trace.shape[0])
        d_traces.append(d_trace)
    return np.array(d_traces)

def _inkml_to_image(inkml_file):
    traces = _parse_traces(inkml_file)
    image_id = _parse_meta(inkml_file)
    coords = []
    for trace in traces:
        coords.extend(trace)
    coords = np.array(coords)
    return image_id, _coords_to_image(coords,False) 


def _size_normalize(images):
    norm_images = np.zeros((images.shape[0], NIS[0], NIS[1]))
    for i, image in enumerate(images):
        print(image.shape)
        norm_image = transform.resize(image, NIS)
        norm_images[i, :, :] = norm_image
    return norm_images

def _pp_common(images):
    images = _size_normalize(images)
    X = image_to_input(images)
    return X


def _pp_chorme(inkml_file):
    image = _inkml_to_image(inkml_file)
    return _pp_common(image)


def _pp_minist(data):
    images = np.reshape(data, (-1,28,28))
    images = images / 255
    return _pp_common(images)


######################################
# API
######################################

# Normalized image shape
NIS = (36, 36)
PROCESSING_FUNC = {
    'MINIST': _pp_minist,
    'CHORME': _pp_chorme
}


def preprocess(data, dtype="MINIST"):
    if(dtype not in PROCESSING_FUNC):
        raise Exception("Fuck you!")
    func = PROCESSING_FUNC[dtype]
    return func(data)


def image_to_input(image):
    if(len(image.shape) == 2):
        image = np.expand_dims(image, axis=0)
    return np.reshape(image, (image.shape[0], image.shape[1]*image.shape[2]))


def input_to_image(X):
    if(len(X.shape) == 2):
        X = np.exp(X, axis=0)
    return np.reshape(X, (X.shape[0], NIS[0], NIS[1]))

