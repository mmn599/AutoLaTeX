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
from skimage import feature
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


DEFAULT_IMAGE_SIZE = (50,50)
def preprocess_image(iraw, ft=None, fthog=None):
    '''
    Converts images (grayed, squared symbol images) into data for learning model
    '''
    iprocessed = iraw.copy()
    # iprocessed = filters.rank.mean(iprocessed, morphology.disk(5))
    # iprocessed = filters.gaussian(iraw, 2)
    iprocessed = resize(iprocessed, DEFAULT_IMAGE_SIZE)
    return iprocessed


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


def preprocess_and_get_inputs(irawsymbols, ft=None, fthog=None):
    input_images = []
    input_hogs = []
    images_processed = []
    images_hog = []

    for iraw in irawsymbols:
        iprocessed = preprocess_image(iraw)
        input_image = iprocessed.flatten()
        input_hog, ihog = feature.hog(iprocessed, visualise=True)

        input_images.append(input_image)
        input_hogs.append(input_hog)
        images_processed.append(iprocessed)
        images_hog.append(ihog)

    input_images = np.array(input_images)
    input_images = np.squeeze(input_images)
    input_hogs = np.array(input_hogs)
    input_hogs = np.squeeze(input_hogs)

    if(ft):
        input_images = ft.transform(input_images)
    if(fthog):
        input_hogs = fthog.transform(input_hogs)

    return input_images, input_hogs, images_processed, images_hog


def predict(input_images, input_hogs, clf, clfhog, ft, fthog):
    ypred = clf.predict(input_images)
    ypredhog = clfhog.predict(input_hogs)
    return ypredhog


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


FN_ILABELS = "current_ilabels"
FN_ISYMBOL = "current_symbol"
DIRME = os.path.dirname(os.path.realpath(__file__))
DIRTEMP = DIRME + "/" + "temp"
DIRMODELS = DIRME + '/models/'
FNCLF = DIRMODELS + 'Model' + '.p'
FNTS = DIRMODELS + 'Ft' + '.p'
CLF = joblib.load(FNCLF)
FT = joblib.load(FNTS)
FNCLFHOG = DIRMODELS + 'ModelHog' + '.p'
FNFTHOG = DIRMODELS + 'FtHog' + '.p'
CLFHOG = joblib.load(FNCLFHOG)
FTHOG = joblib.load(FNFTHOG)


def do_the_damn_thing(fnimage, count):
    ilabels, irawsymbols = file_to_raw_symbols(fnimage)
    input_images, input_hogs, images_processed, images_hog = preprocess_and_get_inputs(irawsymbols, FT, FTHOG)
    ypred = predict(input_images, input_hogs, CLF, CLFHOG, FT, FTHOG)
    latex = prediction_to_latex(ypred)

    fn_ilabels = DIRTEMP + "/" + FN_ILABELS + count + ".png"
    plt.imsave(fn_ilabels, ilabels)

    fns_ips = []
    for i, iprocessedsymbol in enumerate(images_processed):
        fn_ips = DIRTEMP + "/" + FN_ISYMBOL + count + "_" + str(i) + ".png"
        plt.imsave(fn_ips, iprocessedsymbol, cmap="Greys_r")
        fns_ips.append(fn_ips)

    return latex, fn_ilabels, fns_ips, irawsymbols, images_processed, 


if __name__ == "__main__":
    imagefn = sys.argv[1]
    count = sys.argv[2]

    latex, fn_ilabels, fns_ips, irawsymbols, images_processed = do_the_damn_thing(imagefn, count)
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

