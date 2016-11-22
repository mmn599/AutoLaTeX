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
warnings.filterwarnings("ignore")

DEFAULT_IMAGE_SIZE = (28,28)

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


def find_symbol(image):
    image = 1 - image
    nz = np.nonzero(image)
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

def preprocess(image, image_size, ft):
    image = color.rgb2gray(image)
    rowmin, rowmax, colmin, colmax = find_symbol(image)
    image = image[rowmin:rowmax+1, colmin:colmax+1]
    image = square_image(image)
    image = morphology.erosion(image, morphology.square(5))
    image = filters.rank.median(image, morphology.square(3))
    image = resize(image, image_size)
    image_input = image.flatten()
    if(len(image_input.shape)==1):
        image_input = np.expand_dims(image_input, axis=0)
    if(ft):
        image_input = ft.transform(image_input)
    return image, image_input


def predict(image, clf, ft):
    image, image_input = preprocess(image, (28,28), ft)
    return clf.predict(image_input)


def prediction_to_latex(prediction):
    return prediction[0]


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
    image = io.imread(imagefn)
    clf = joblib.load(clfloc)
    ft = joblib.load(ftloc)
    prediction = predict(image, clf, ft)
    latex = prediction_to_latex(prediction)
    sys.stdout.write(latex)

