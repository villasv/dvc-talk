import pathlib
import sys
from joblib import load
import pandas as pd

splits = pathlib.Path(sys.argv[1])
test_images = pd.read_csv(splits / "test_images.csv")
test_labels = pd.read_csv(splits / "test_labels.csv")
clf = load(sys.argv[2])

mean_accuracy = clf.score(test_images,test_labels)

with open(sys.argv[3], 'w') as fd:
    fd.write('MeanAcc: {:4f}\n'.format(mean_accuracy))
