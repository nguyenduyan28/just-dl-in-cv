import numpy as np
from load_ds import load_cifar10



def add_ones(M):
  if (M.ndim == 1):
    M = M.reshape(-1, 1)
  ones = np.ones((1, M.shape[1]))
  return np.vstack((M, ones))


def loss_SVM(delta, classes : np.array, y_idx):
  '''
  Suppose classes is 1D array class contains points
  '''
  class_points = classes - classes[y_idx] + delta
  class_points[y_idx] = 0
  return np.sum(np.maximum(0, class_points))

def loss_SVM_regularization(delta, classes : np.array, y_idx, W : np.array, lamb):
  L_i = loss_SVM(delta, classes, y_idx) 
  R = np.sum(W * W)
  return L_i + lamb * R


def L(X, y, W, delta=1.0):
  """
  X: shape [D x N] - mỗi ảnh là 1 cột
  y: shape [N] - chỉ số lớp đúng
  W: shape [C x D]
  """
  N = X.shape[1]
  scores = (W @ X).T # 
  correct_class_score  = scores[np.arange(X.shape[1]), y]
  L_i = scores  - correct_class_score[:, np.newaxis] + delta
  L_i[np.arange(X.shape[1]), y] = 0
  L_i = np.sum(np.maximum(0, L_i)) / N
  return L_i
  
  
def softmax_function(f):
  f = f - np.max(f)
  f_exp = np.exp(f)
  sum_f = np.sum(f_exp)
  return f_exp / sum_f


def softmax_function_batch(F):
  '''
  F is now shape [N x C]
  '''
  F = F - np.max(F, axis = 1, keepdims=True)
  F_exp = np.exp(F)
  sum_F = np.sum(F_exp, axis=1, keepdims=True)
  return F_exp / sum_F  
    

def CrossEntropyLoss(F, y):
  F = F - np.max(F, axis = 1, keepdims=True)
  correct_class_score = F[np.arange(F.shape[0]), y]
  log_sum_other = np.log(np.sum(np.exp(F), axis=1, keepdims=True))
  print(correct_class_score[:, np.newaxis] - log_sum_other)
  return (-1 / F.shape[0]) * np.sum(correct_class_score[:, np.newaxis] - log_sum_other) 

def score_calc(x_i, num_of_class = 10):
  '''
  X_i will have shape : 3072,
  '''
  x_i = add_ones(x_i)
  W = np.random.randn(x_i.shape[0], num_of_class)
  print(W.shape)
  return x_i.T @ W 
  
def test():
  F = np.array([[2.0, 1.0, 0.1],
                [1.0, 2.0, 3.0]])
  y = np.array([0, 1])
  print(CrossEntropyLoss(F,y))

if __name__ == '__main__':
  test()