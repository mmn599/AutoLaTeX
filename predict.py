import preprocessing as pp
import sys
from skimage import io
from sklearn.externals import joblib
import os

def predict(data, clf, dtype="APP"):
    data = pp.preprocess(data, dtype)
    return clf.predict(data)


def prediction_to_latex(prediction):
	return "\\int a^2dx"


if __name__ == "__main__":
	fn = sys.argv[1]
	clfloc = sys.argv[2]
	data = io.imread(fn)
	scriptdir = os.getcwd()
	clf = joblib.load(clfloc)
	prediction = predict(data, clf, dtype="APP")
	latex = prediction_to_latex(prediction)
	sys.stdout.write(latex)

