import sys
from skimage import io
from skimage import morphology
from skimage import color
import numpy as np
import warnings
import skimage
import glob
from skimage import measure
from skimage import feature
from sklearn.externals import joblib
from matplotlib import pyplot as plt
import os
warnings.filterwarnings("ignore")

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
    'rightparen' : ')',
    'sum':'\\sum ',
    'v' : 'v',
    'ww' : 'W',
    'divslash' : '/',
    'u' : 'u',
    'ii' : 'I',
    'comma' : ','
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


def myresize(images, image_size):
    inp_processedz = np.zeros((len(images), image_size[0] * image_size[1]))
    iprocessedz = np.zeros((len(images), image_size[0], image_size[1]))
    for i, image in enumerate(images):
        image = image.squeeze()
        image = skimage.filters.gaussian(image, 1)
        image = skimage.transform.resize(image, image_size)
        iprocessedz[i, :, :] = image
        inp_processedz[i, :] = image.flatten()
    return inp_processedz, iprocessedz


DEFAULT_IMAGE_SIZE = (40,40)


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


def thresh_label(image, label):
    image_th = image.copy()
    image_th[image == label] = 1
    image_th[image != label] = 0
    return image_th


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


def find_closest(ilabels, smalllabel):
    ithresh = thresh_label(ilabels, smalllabel)
    rowmin, rowmax, colmin, colmax = find_symbol(ithresh)
    iselect = ilabels[rowmax:ilabels.shape[0],colmin - 10 : colmax + 10] 
    iselect[iselect == smalllabel] = 0
    newlabel = np.max(iselect)
    return newlabel


def seperate_symbols(overall_image):
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
    labels = []

    # Find the large labels
    for label in range(1, np.max(ilabels)+1):
        label_th = thresh_label(ilabels, label)
        count = np.count_nonzero(label_th)
        if(count > 50):
            labels.append(label)
        else:
            newlabel = find_closest(ilabels, label)
            ilabels[ilabels == label] = newlabel

    if(len(labels) == 0):
        raise Exception('Bad image!')

    for label in labels:
        label_th = thresh_label(ilabels, label)
        rowmin, rowmax, colmin, colmax = find_symbol(label_th)
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
    return ilabels, np.array(square_symbols)


def preprocess_images(images):
    image_size = (40,40)
    X, Ximages = myresize(images, image_size)

    shape = len(skimage.feature.hog(images[0]))
    Xt = np.zeros((len(images), shape))
    images_hog = []
    for i, image in enumerate(images):
        Xt[i, :], ihog = skimage.feature.hog(image, visualise=True)
        images_hog.append(ihog)

    images_hog = np.array(images_hog)

    return Xt, images_hog


def prediction_to_latex(predictions):
    latex = ""
    for pred in predictions:
        latex = latex + LATEX_BOOK[pred]
    return latex


def file_to_raw_symbols(fn, single_symbol=False):
    ioverall = io.imread(fn)
    ilabels, irawsymbols = seperate_symbols(ioverall)
    return ilabels, irawsymbols
     

def get_custom_data(datadir):
    '''
    Returns input_images, input_hogs, symbols, images_processed, images_hog, images_raw_symbols
    '''
    images_raw_symbols = []
    symbols = []
    for name in glob.glob(datadir + '*.png'):
        symbol = name.split('_')[1].replace(".png","")
        try:
            ilabels, irawsymbols = file_to_raw_symbols(name, True)
        except:
            print(name) 
        if(len(irawsymbols)!=1):
            raise Exception('More than one symbol found in training data!')
        images_raw_symbols.append(irawsymbols[0])
        symbols.append(symbol)
    input_images, input_hogs, images_processed, images_hog = preprocess_and_get_inputs(images_raw_symbols)
    return input_images, input_hogs, images_processed, images_hog, symbols, images_raw_symbols


DIRTEMP = "C:\\Users\\mmnor\\Projects\\autolatex\\temp"

def do_the_damn_thing(fnimage, fnmodel, count):
    ilabels, irawsymbols = file_to_raw_symbols(fnimage)
    Xt, images_hog = preprocess_images(irawsymbols)
    model = joblib.load(fnmodel)
    ypred = model.predict(Xt)
    latex = prediction_to_latex(ypred)

    fn_ilabels = DIRTEMP + "/" + "current_ilabels" + count + ".png"
    plt.imsave(fn_ilabels, ilabels)

    fns_ips = []
    for i, iprocessedsymbol in enumerate(images_processed):
        fn_ips = DIRTEMP + "/" + "current_symbol" + count + "_" + str(i) + ".png"
        plt.imsave(fn_ips, iprocessedsymbol, cmap="Greys_r")
        fns_ips.append(fn_ips)

    return latex, fn_ilabels, fns_ips, irawsymbols, images_processed, 


if __name__ == "__main__":
    imagefn = sys.argv[1]
    count = sys.argv[2]

    fnmodel = "models/Pipe.p"
    
    latex, fn_ilabels, fns_ips, irawsymbols, images_processed = do_the_damn_thing(imagefn, fnmodel, count)
    numsymbols = len(fns_ips)

    sys.stdout.write(latex)
    sys.stdout.write('\n')
    sys.stdout.write(fn_ilabels)
    sys.stdout.write('\n')
    sys.stdout.write(str(numsymbols))
    sys.stdout.write('\n')
    for fn in fns_ips:
        sys.stdout.write(fn)
        sys.stdout.write('\n')
    sys.stdout.flush()

