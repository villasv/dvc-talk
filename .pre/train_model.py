import pathlib
import sys

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from joblib import dump

splits = pathlib.Path(sys.argv[1])
train_images = pd.read_csv(splits / "train_images.csv")
train_labels = pd.read_csv(splits / "train_labels.csv")

# train_images[train_images>0]=1

clf = svm.SVC()
clf.fit(train_images, train_labels.values.ravel())
dump(clf, sys.argv[2])
