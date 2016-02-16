import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

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
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_classes = W.shape[1]
  
  for i in xrange(num_train):
    score = X[i,:].dot(W)
    score -= np.max(score)    # shift to avoid numerical unstability
    exp_score = np.exp(score)
    prob = exp_score / np.sum(exp_score)   # normalization to get the prob    
    loss += -np.log(prob[y[i]])
    for j in xrange(num_classes):
      if j == y[i]:          # dL/dP_k = P_k-1(k == yi), dP_k/dW_k = X'
        dW[:, j] += (prob[j]-1)*(X[i, :].T)  
      else:
        dW[:, j] += prob[j]*(X[i, :].T)
    
  loss /= num_train
  loss += 0.5 * reg * np.sum(W*W)
  dW /= num_train
  dW += reg * W  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_classes = W.shape[1]  
  score = X.dot(W)
  score -= np.amax(score, axis = 1, keepdims = True)  # keepdims = True for broadcasting
  exp_score = np.exp(score)
  prob = exp_score/np.sum(exp_score, axis = 1, keepdims = True)
  loss = -np.log(prob[range(num_train), y]).sum()
  
  loss /= num_train
  loss += 0.5 * reg * np.sum(W*W)
  
  #############################################################################
  # dL/dP_k = P_k-1(k == yi), dP_k/dW_k = X', chain rule!
  dprob = prob
  dprob[range(num_train), y] -= 1
  dW = X.T.dot(dprob)
  
  dW /= num_train
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

