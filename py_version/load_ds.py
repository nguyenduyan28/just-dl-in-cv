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





def reformat_arr(img_arr):
  r = img_arr[: , : 1024].reshape(-1, 1)
  g = img_arr[:, 1024 : 1024 * 2].reshape(-1, 1)
  b = img_arr[:, 1024 * 2 :].reshape(-1, 1)
  img_new = np.concatenate((r, g, b), axis = 1).reshape(img_arr.shape[0], -1)
  return img_new


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

  X_train = reformat_arr(np.concatenate(X_train).reshape(-1, 3072))
  y_train = np.array(y_train).reshape(-1, 1)
  X_test = reformat_arr(np.concatenate(X_test).reshape(-1, 3072))
  y_test = np.array(y_test).reshape(-1, 1)
  if val_split:
    X_val = X_train[-10000 :]
    y_val = y_train[-10000 :]
    X_train = X_train[: -10000]
    y_train = y_train[: -10000]
    return (X_train), (y_train), (X_val), (y_val), (X_test), (y_test)
  return (X_train), (y_train), (X_test), (y_test)


def show_img(img_arr, label_arr):
  img_arr_normalized = img_arr.reshape(32, 32, 3) # transpose to switch C, H, W to W, H, C, reshape here because cifar10 rgb 
  label_list = cifar10_get_labels()
  label_name = label_list[label_arr[0]]
  plt.title(label_name)
  plt.imshow(img_arr_normalized) 
  plt.show()
