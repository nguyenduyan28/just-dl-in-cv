import urllib
import requests
import tarfile
import os
import numpy as np
from matplotlib import pyplot as plt
import pickle

DATA_PATH = '/Users/duyan/datasets/cifar-10-batches-py'
def extract_cifar10():
  url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
  response = requests.get(url, stream=True)
  file = tarfile.open(fileobj=response.raw, mode="r|gz")
  file.extractall('/Users/duyan/datasets')


def unpickle(file):
  with open(file, 'rb') as fo:
    dict = pickle.load(fo, encoding='bytes')
  return dict

def cifar10_get_labels():
  for root, dirs, files in os.walk(DATA_PATH):
    for file in files:
      if ("batches.meta" in file):
        file = os.path.join(root, file)
        raw_data = unpickle(file)
        label_names = [data.decode() for data in raw_data[b'label_names']]
        return label_names



def load_cifar10(val_split = False, return_label = False):
  X_train = [] 
  y_train = []
  X_test = []
  y_test = []
  for root, dirs, files in os.walk(DATA_PATH):
    for file in files:
      if ("batch" in file):
        file = os.path.join(root, file)
        raw_data = unpickle(file)
        if ("data_batch" in file):
          X_train.append(raw_data[b'data'])
          y_train.extend(raw_data[b'labels'])
        elif("test_batch" in file):
          X_test.append(raw_data[b'data'])
          y_test.extend(raw_data[b'labels'])
        elif ("batches.meta" in file):
          label_names = [data.decode() for data in raw_data[b'label_names']]

  X_train = np.concatenate(X_train).reshape(-1, 3072)
  y_train = np.array(y_train).reshape(-1, 1)
  X_test = np.concatenate(X_test).reshape(-1, 3072)
  y_test = np.array(y_test).reshape(-1, 1)
  if val_split:
    X_val = X_train[-10000 :]
    y_val = y_train[-10000 :]
    X_train = X_train[: -10000]
    y_train = y_train[: -10000]
    return (X_train), (y_train), (X_val), (y_val), (X_test), (y_test)
  return (X_train), (y_train), (X_test), (y_test)


def show_img(img_arr):
  img_arr = img_arr.reshape(3, 32, 32).transpose(1, 2, 0) # transpose to switch C, H, W to W, H, C, reshape here because cifar10 rgb 
  print(img_arr.dtype)
  img_arr_normalized = np.clip(img_arr, 0, 255).astype(np.uint8) # if any < 0 --> 0, any > 255 --> 255
  plt.imshow(img_arr_normalized) 
  plt.show()







if __name__ == '__main__':
  X_train, y_train, X_val, y_val, X_test, y_test = load_cifar10(val_split=True)
  show_img(X_train[1])