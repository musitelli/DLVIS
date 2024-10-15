from builtins import range
import numpy as np



def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # batch size = amount of examples N (minibatch)
    batch_size = x.shape[0]

    # number of features = d_1 * d_2 * ... * d_k = np.prod(d_1, d_2, ... , d_k) = np.prod(x.shape[1:])
    num_features = np.prod(x.shape[1:])

    # x must be reshaped since the ANN expects a 2D matrix as an input
    x_reshaped = x.reshape(batch_size, num_features)

    # Affine transformation is computed below
    out = x_reshaped.dot(w) + b

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # As in last section

    # batch size = amount of examples N (minibatch)
    batch_size = x.shape[0]

    # number of features = d_1 * d_2 * ... * d_k = np.prod(d_1, d_2, ... , d_k) = np.prod(x.shape[1:])
    num_features = np.prod(x.shape[1:])

    # Remember: out = x_reshaped.dot(w) + b

    # dx = dout * w
    dx = dout.dot(w.T)

    # Since dx has shape = (N,D), must be reshaped to shape = (N, d_1, ... d_k)
    dx = dx.reshape(x.shape)

    # dw = x_reshaped * dout
    dw = x.reshape(batch_size, num_features).T.dot(dout)

    # db = 1s * dout
    db = np.sum(dout,axis=0)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    out = np.maximum(0,x)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # (ReLu(x))' = 1 if x > 0, else 0

    # Hence the chain rule applies the element-wise product of the matrices x>0 and dout

    # (x>0) is the ReLu : returns True ~ 1 if positive and False ~ 0 if negative

    dx = (x>0) * dout

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    loss, dx = None, None

    ###########################################################################
    # TODO: Copy over your solution from A1.
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Scores, as before
    s = x

    # Stabilize by subtracting the maximum value per row for numerical stability
    s = s - np.max(s, axis=1, keepdims=True)

    # Correct class scores s_y (extract the scores corresponding to the correct class labels)
    s_y = s[range(s.shape[0]), y]

    # Loss
    loss = -np.mean(np.log(np.exp(s_y) / np.sum(np.exp(s), axis=1)))

    # Init. matrix to be the ones matrix
    ones = np.zeros(s.shape)

    # Convert by the identity function (one-hot encode labels)
    ones[range(s.shape[0]), y] = 1

    # Probability matrix
    P = np.exp(s) / np.sum(np.exp(s), axis=1, keepdims=True)

    # Gradient with respect to x (the other implementation computed dW)
    dx = (P - ones) / x.shape[0]

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return loss, dx
