import sys
from skimage import io
from sklearn.externals import joblib
from skimage.transform import resize
from skimage import morphology
from skimage import color
import numpy as np
import warnings
from skimage import filters
import glob
from skimage import feature
from scipy import ndimage as ndi
warnings.filterwarnings("ignore")

DEFAULT_IMAGE_SIZE = (28,28)
LATEX_BOOK = {
    'a' : 'a',
    'b' : 'b',
    'c' : 'c',
    'd' : 'd',
    'e' : 'e',
    'i' : 'i',
    'j' : 'j',
    'q' : 'q',
    'x' : 'x',
    'nn' : 'N',
    'mm' : 'M',
    '0' : '0',
    '1' : '1',
    '2' : '2',
    '3' : '3',
    '4' : '4',
    '5' : '5',
    '6' : '6',
    '7' : '7',
    '8' : '8',
    '9' : '9',
    'plus' : '+',
    'minus' : '-',
    'int' : '\\int',
    'leftparen' : '(',
    'rightparen' : ')'
}

def find_average_size(images):
    Ms = []
    Ns = []
    for image in images:
        Ms.append(image.shape[0])
        Ns.append(image.shape[1])
    Ms = np.array(Ms)
    Ns = np.array(Ns)
    m = np.ceil(np.mean(Ms))
    n = np.ceil(np.mean(Ns))
    return (m, n)


def preprocess(image, image_size, ft):
    '''
    Converts images (grayed, squared symbol images) into data for learning model
    '''
    image = morphology.erosion(image, morphology.square(5))
    image = filters.rank.median(image, morphology.square(3))
    image = resize(image, image_size)
    image_input = image.flatten()
    if(len(image_input.shape)==1):
        image_input = np.expand_dims(image_input, axis=0)
    if(ft):
        image_input = ft.transform(image_input)
    return image, image_input


def find_label(image, label):
    image_th = image.copy()
    image_th[image == label] = 1
    image_th[image != label] = 0
    nz = np.nonzero(image_th)
    rowmin = np.min(nz[0])
    rowmax = np.max(nz[0])
    colmin = np.min(nz[1])
    colmax = np.max(nz[1])
    return rowmin, rowmax, colmin, colmax


def square_image(image):
    height = image.shape[0]
    width = image.shape[1]
    themax = max(height, width)
    newimage = np.ones((themax, themax))
    if(height > width):
        diff = (height - width)//2
        newimage[:,diff:diff+width] = image
    else:
        diff = (width - height)/2
        newimage[diff:diff+height, :] = image
    return newimage


def overall_to_symbols(overall_image):
    '''
    Returns all squared, grayed, symbols
    '''
    overall_image = color.rgb2gray(overall_image)
    o_image = overall_image.copy()
    o_image[overall_image < 1] = 1
    o_image[overall_image == 1] = 0
    edges = feature.canny(o_image)
    fill_edges = ndi.binary_fill_holes(edges)
    label_objects, nb_labels = ndi.label(fill_edges)
    symbols = []
    for label in range (1, nb_labels+1):
        rowmin, rowmax, colmin, colmax = find_label(label_objects, label)
        isymbol = overall_image[rowmin:rowmax+1,colmin:colmax+1]
        symbols.append((colmin, isymbol))
    sorted_by_colmin = sorted(symbols, key=lambda tup: tup[0])
    symbols = [el[1] for el in sorted_by_colmin]
    symbols = np.array(symbols)
    square_symbols = []
    for symbol in symbols:
        ss = square_image(symbol)
        square_symbols.append(ss)
    return np.array(square_symbols)


def predict(image_overall, clf, ft):
    images = overall_to_symbols(image_overall)
    X = []
    for i, image in enumerate(images):
        image, image_input = preprocess(image, DEFAULT_IMAGE_SIZE, ft)
        X.append(image_input)
    X = np.array(X)
    X = np.squeeze(X)
    if(len(X.shape)==1):
        X = np.expand_dims(X, axis=0)
    return list(clf.predict(X))


def prediction_to_latex(predictions):
    latex = ""
    for pred in predictions:
        latex = latex + LATEX_BOOK[pred]
    return latex


def get_custom_data(datadir, n_image_size=None):
    images = []
    symbols = []

    for name in glob.glob(datadir + '*.png'):
        symbol = name.split('_')[1].replace(".png","")
        image = io.imread(name)
        images.append(image)
        symbols.append(symbol)

    images = np.array(images)

    X = []
    X_images = []
    for image in images:
        image, image_input = preprocess(image, n_image_size, None)
        X.append(image_input)
        X_images.append(image)
    X = np.array(X)
    X = np.squeeze(X)
        
    return X_images, X, symbols, n_image_size

if __name__ == "__main__":
    imagefn = sys.argv[1]
    clfloc = sys.argv[2]
    ftloc = sys.argv[3]
    image_overall = io.imread(imagefn)
    clf = joblib.load(clfloc)
    ft = joblib.load(ftloc)
    prediction = predict(image_overall, clf, ft)
    latex = prediction_to_latex(prediction)
    sys.stdout.write(latex)

