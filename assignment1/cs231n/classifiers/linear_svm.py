import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape, dtype = 'float64') # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  # dW = \sum dW[i]
  for i in xrange(num_train):
    cnt = 0    # the count of (delta_score + delta_margin > 0), see Line 35
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        cnt += 1
        loss += margin
        dW[:, j] += X[i]
    dW[:, y[i]] += -cnt*X[i]
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train
  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg*W   # f(W) = 0.5*reg*W*WT 
  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################
  # see above
  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape, dtype = 'float64') # initialize the gradient as zero
  # print dW.shape
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  num_train = X.shape[0]
  scores = X.dot(W)
  correct_class_scores = scores[range(0, num_train), y].reshape(num_train, -1)
  #print scores.shape
  #print correct_class_scores.shape  
  margin = scores - correct_class_scores + 1   # the margin is 1
  margin[range(0, num_train), y] = 0    # when j = yi, continue
  margin = np.maximum(np.zeros(margin.shape), margin)  
  
  loss = np.sum(margin) / num_train 
  loss += 0.5 * reg * np.sum(W * W)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  num_loss = (margin > 0).sum(1)   # the number of (L > 0) for each train sample
  margin[margin > 0] = 1;
  margin[range(0, num_train), y] = -num_loss;
  dW = X.T.dot(margin) / num_train + reg*W;
  #############################################################################
  # 
  # X' = [X_1' X_2' ... X_N'] 
  # margin contains the weights that each X_i contributes to the final gradient
  # try to understand the vectorization using the thought of "Partitioned Matrix"
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
