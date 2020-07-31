import pathlib
import sys
import pandas as pd
from sklearn.model_selection import train_test_split


labeled_images = pd.read_csv(sys.argv[1])

images = labeled_images.iloc[0:5000, 1:]
labels = labeled_images.iloc[0:5000, :1]

train_images, test_images, train_labels, test_labels = train_test_split(
    images, labels, train_size=0.8, random_state=0,
)

splits = pathlib.Path(sys.argv[2])
splits.mkdir(parents=True, exist_ok=True)

train_images.to_csv(splits / "train_images.csv", index=False)
train_labels.to_csv(splits / "train_labels.csv", index=False)
test_images.to_csv(splits / "test_images.csv", index=False)
test_labels.to_csv(splits / "test_labels.csv", index=False)
