import preprocessing as pp

def predict(data, clf, dtype="APP"):
    data = pp.preprocess(data, dtype)
    return clf.predict(data) 