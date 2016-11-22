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
from skimage import measure
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
    'int' : '\\int ',
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


def find_symbol(image):
    '''
    Assumes background of 0 and objects closer to 1
    '''
    nz = np.nonzero(image)
    rowmin = np.min(nz[0])
    rowmax = np.max(nz[0])
    colmin = np.min(nz[1])
    colmax = np.max(nz[1])
    return rowmin, rowmax, colmin, colmax


def find_label(image, label):
    image_th = image.copy()
    image_th[image == label] = 1
    image_th[image != label] = 0
    return find_symbol(image_th)


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
    ithresh = overall_image.copy()
    ithresh[overall_image < 1] = 1
    ithresh[overall_image == 1] = 0
    ithresh = morphology.closing(ithresh, morphology.disk(5))
    ilabels = measure.label(ithresh, background=0)
    symbols = []
    for label in range (1, np.max(ilabels)+1):
        rowmin, rowmax, colmin, colmax = find_label(ilabels, label)
        isymbol = overall_image[rowmin:rowmax+1,colmin:colmax+1]
        srowmin, srowmax, scolmin, scolmax = find_symbol(1 - isymbol)
        isymbol = isymbol[srowmin:srowmax + 1, scolmin:scolmax+1]
        symbols.append((colmin, isymbol))
    sorted_by_colmin = sorted(symbols, key=lambda tup: tup[0])
    symbols = [el[1] for el in sorted_by_colmin]
    symbols = np.array(symbols)
    square_symbols = []
    for symbol in symbols:
        ss = square_image(symbol)
        square_symbols.append(ss)
    return np.array(square_symbols)


def predict(irawsymbols, clf, ft):
    X = []
    for i, image in enumerate(irawsymbols):
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


def file_to_raw_symbols(fn, single_symbol=False):
    ioverall = io.imread(fn)
    try:
        irawsymbols = overall_to_symbols(ioverall)
    except:
        raise Exception(fn) 
    if(single_symbol and len(irawsymbols) != 1):
        raise Exception(fn)
    return irawsymbols
     

def get_custom_data(datadir, n_image_size=None):
    y = []
    X = []
    processed_images = []
    original_images = []

    for name in glob.glob(datadir + '*.png'):
        symbol = name.split('_')[1].replace(".png","")
        irawsymbols = file_to_raw_symbols(name, (28,28), True)
        iprocessed, image_input = preprocess(irawsymbols[0], None)
        processed_images.append(iprocessed)
        original_images.append(irawsymbols[0])
        X.append(image_input)
        y.append(symbol)

    X = np.array(X)
    X = np.squeeze(X)
        
    return X, y, processed_images, original_images

if __name__ == "__main__":
    imagefn = sys.argv[1]
    clfloc = sys.argv[2]
    ftloc = sys.argv[3]
    clf = joblib.load(clfloc)
    ft = joblib.load(ftloc)
    irawsymbols = file_to_raw_symbols(imagefn)
    prediction = predict(irawsymbols, clf, ft)
    latex = prediction_to_latex(prediction)
    sys.stdout.flush()
    sys.stdout.write(latex)

