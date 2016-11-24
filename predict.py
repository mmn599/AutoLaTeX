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
from matplotlib import pyplot as plt
import os
warnings.filterwarnings("ignore")

DEFAULT_IMAGE_SIZE = (36,36)
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


def pre_v4(image):
    image_size = (50, 50)
    image = filters.rank.mean(image, morphology.disk(1))
    image = filters.gaussian(image, sigma=.5)
    image = resize(image, image_size)
    ibin = image.copy()
    ibin[image < 1] = 1
    ibin[image == 1] = 0
    return ibin


def pre_v3(image):
    image_size = (50, 50)
    image = filters.rank.mean(image, morphology.disk(3))
    image = resize(image, image_size)
    return image


def pre_v2(image):
    image_size = (50, 50)
    image = resize(image, image_size)
    return image


def pre_v1(image):
    image_size = (36, 36)
    image = filters.rank.mean(image, morphology.disk(3))
    image = resize(image, image_size)
    return image


def preprocess(iraw, version=1, ft=None):
    '''
    Converts images (grayed, squared symbol images) into data for learning model
    '''
    # image = morphology.erosion(image, morphology.square(5))
    # image = filters.rank.median(image, morphology.square(3))
    version = int(version)
    if(version==1):
        iprocessed = pre_v1(iraw)
    elif(version==2):
        iprocessed = pre_v2(iraw)
    elif(version==3):
        iprocessed = pre_v3(iraw)
    elif(version==4):
        iprocessed = pre_v4(iraw)
    else:
        raise Exception('Wrong version: ' + str(version))
    image_input = iprocessed.flatten()
    if(len(image_input.shape)==1):
        image_input = np.expand_dims(image_input, axis=0)
    if(ft):
        image_input = ft.transform(image_input)
    return iprocessed, image_input


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


def predict(irawsymbols, clf, ft, version):
    X = []
    images_processed = []
    for iraw in irawsymbols:
        iprocessed, image_input = preprocess(iraw, version, ft)
        X.append(image_input)
        images_processed.append(iprocessed)
    X = np.array(X)
    X = np.squeeze(X)
    if(len(X.shape)==1):
        X = np.expand_dims(X, axis=0)
    ypred = clf.predict(X)
    return images_processed, X, ypred


def prediction_to_latex(predictions):
    latex = ""
    for pred in predictions:
        latex = latex + LATEX_BOOK[pred]
    return latex


def file_to_raw_symbols(fn, single_symbol=False):
    ioverall = io.imread(fn)
    try:
        ilabels, irawsymbols = seperate_symbols(ioverall)
    except:
        raise Exception(fn) 
    if(single_symbol and len(irawsymbols) != 1):
        raise Exception(fn)
    return ilabels, irawsymbols
     

def get_custom_data(datadir, version):
    y = []
    X = []
    processed_images = []
    original_images = []

    for name in glob.glob(datadir + '*.png'):
        symbol = name.split('_')[1].replace(".png","")
        irawsymbols = file_to_raw_symbols(name, True)
        iprocessed, image_input = preprocess(irawsymbols[0], version, None)
        processed_images.append(iprocessed)
        original_images.append(irawsymbols[0])
        X.append(image_input)
        y.append(symbol)

    X = np.array(X)
    X = np.squeeze(X)
        
    return X, y, processed_images, original_images

FN_ILABELS = "current_ilabels"
FN_ISYMBOL = "current_symbol"

def do_the_damn_thing(fnimage, version, count):
    dirme = os.path.dirname(os.path.realpath(__file__))
    dirtemp = dirme + "/" + "temp"
    dirmodels = dirme + '/models/'
    fnclf = dirmodels + 'Model' + str(version) + '.p'
    fnts = dirmodels + 'Ts' + str(version) + '.p'
    clf = joblib.load(fnclf)
    ts = joblib.load(fnts)

    ilabels, irawsymbols = file_to_raw_symbols(fnimage)
    images_processed, X, ypred = predict(irawsymbols, clf, ts, version=version)
    latex = prediction_to_latex(ypred)

    fn_ilabels = dirtemp + "/" + FN_ILABELS + count + ".png"
    plt.imsave(fn_ilabels, ilabels)

    fns_ips = []
    for i, iprocessedsymbol in enumerate(images_processed):
        fn_ips = dirtemp + "/" + FN_ISYMBOL + count + "_" + str(i) + ".png"
        plt.imsave(fn_ips, iprocessedsymbol, cmap="Greys_r")
        fns_ips.append(fn_ips)

    return latex, fn_ilabels, fns_ips, irawsymbols, images_processed, 


if __name__ == "__main__":
    imagefn = sys.argv[1]
    version = sys.argv[2]
    count = sys.argv[3]

    latex, fn_ilabels, fns_ips, irawsymbols, images_processed = do_the_damn_thing(imagefn, version, count)
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

