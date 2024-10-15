from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    for n in range(X.shape[0]):
        
        # Compute scores for nth element
        s = X[n].dot(W) 

        # For stability purposes
        s = s-np.max(s)

        # Update the loss (with / X.shape[0] for averaging)
        loss = loss - np.log((np.e ** s[y[n]]) / np.sum(np.e**s) ) / X.shape[0] 

        # Now compute gradientt for each of the classes
        for c in range(W.shape[1]):
            
            # Compute probabilities
            p = (np.e ** s[c]) / np.sum(np.e ** s)

            # Update dW for this specific class c (with / X.shape[0] for averaging)
            dW[:,c] = dW[:,c] + (p - (c==y[n])) * X[n,:] / X.shape[0]
    
    # Include regularization
    loss = loss + reg * np.sum(W**2) # / N
    dW = dW + reg * 2 * W # np.sum(W) # 2 instead of ** 2 because of the derivative

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Scores matrix ; shape s = shape X * shape W = N,D * D,C = N, C
    s = X.dot(W) 

    # Stabilize
    s = s - np.max(s)

    # Correct scores s[n, y[n]]
    s_y = s[range(X.shape[0]),y]

    # Loss (np.mean calculates the same sum as before, the one with / X.shape[0] for averaging)
    loss = -np.mean(np.log(np.e**s_y / np.sum(np.e**s))) + reg * np.sum(W**2)

    # NOTE: the above line looks like P but only for the correct classes (e^y_i)

    # Probability matrix (shape = N, C)
    P = np.e ** s / np.sum(np.e ** s, axis = 1, keepdims=True)

    # Init. matrix to be the ones matrix (shape = N, C)
    ones = np.zeros(s.shape)

    # Convert by the identity function
    ones[range(s.shape[0]),y] = 1

    # Gradient (shape dW = D,C = (N, C .T * N,D).T )
    dW = ((P-ones).T.dot(X)).T / X.shape[0] + reg * 2 * W # np.sum(W) # 2 instead of ** 2 because of the derivative

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
