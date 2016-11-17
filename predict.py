import processing

def predict(data, clf, dtype="Default"):
    data = processing.preprocess(data, dtype)
    return clf.predict(data) 