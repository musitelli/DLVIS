from builtins import range
import numpy as np


def affine_forward(x, w, b):
    """Computes the forward pass for an affine (fully connected) layer.

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
    # TODO: Copy over your solution from Assignment 1.                        #
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
    """Computes the backward pass for an affine (fully connected) layer.

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
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

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
    """Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
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
    """Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
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
    """Computes the loss and gradient for softmax classification.

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
    # TODO: Copy over your solution from Assignment 1.                        #
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


def batchnorm_forward(x, gamma, beta, bn_param):
    """Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param["mode"]
    eps = bn_param.get("eps", 1e-5)
    momentum = bn_param.get("momentum", 0.9)

    N, D = x.shape
    running_mean = bn_param.get("running_mean", np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get("running_var", np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == "train":
        #######################################################################
        # Read carefully each of the steps of the implementation of 
        # training-time forward pass for batch norm.                                          #
        #######################################################################

        # FORWARD PASS: Step-by-Step
        
        # Step 1. m = 1 / N \sum x_i
        m = np.mean(x, axis=0, keepdims=True)
        
        # Step 2. xc = x - m
        xc = x - m
        
        # Step 3. xc2 = xc ^ 2
        xcsq = xc ** 2
        
        # Step 4. v = 1 / N \sum xc2_i
        v = np.mean(xcsq, axis=0, keepdims=True)
        
        # Step 5. vsq = sqrt(v + eps)
        vsqrt = np.sqrt(v + eps)
        
        # Step 6. invv = 1 / vsq
        invv = 1.0 / vsqrt
        
        # Step 7. xn = xc * invv
        xn = xc * invv
        
        # Step 8. xg = xn * gamma
        xgamma = xn * gamma
        
        # Step 9. out = xg + beta
        out = xgamma + beta
        
        cache = (x, xc, vsqrt, v, invv, xn, gamma, eps)
        
        running_mean = momentum * running_mean + (1 - momentum) * m
        running_var = momentum * running_var + (1 - momentum) * v

        #######################################################################
        #                           END OF CODE                               #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # FORWARD PASS: Step-by-Step
        
        # Step 1. xc = x - running_mean
        xc = x - bn_param["running_mean"]
        
        # Step 2. invv = 1 / vsq
        invv = 1.0 / np.sqrt(bn_param["running_var"])
        
        # Step 3. xn = xc * invv
        xn = xc * invv
        
        # Step 4. xg = xn * gamma
        xgamma = xn * gamma
        
        # Step 9. out = xg + beta
        out = xgamma + beta

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param["running_mean"] = running_mean
    bn_param["running_var"] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """Backward pass for batch normalization.
    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # Read carefully each of the steps of the implementation of 
    # training-time backward pass for batch norm.                                              #
    ###########################################################################

    (x, xc, vsqrt, v, invv, xn, gamma, eps) = cache
  
    N, D = x.shape
  
    # BACKWARD PASS: Step-byStep
  
    # Step 9. out = xg + beta
    dxg = dout
    dbeta = np.sum(dout, axis=0)
  
    # Step 8. xg = xn * gamma
    dxn = dxg * gamma
    dgamma = np.sum(dxg * xn, axis=0)
  
    # Step 7. xn = xc * invv
    dxc1 = dxn * invv
    dinvv = np.sum(dxn * xc, axis=0)
  
    # Step 6. invv = 1 / vsqrt
    dvsqrt = -1 / (vsqrt ** 2) * dinvv
  
    # Step 5. vsqrt = sqrt(v + eps)
    dv = 0.5 * dvsqrt / np.sqrt(v + eps)
  
    # Step 4. v = 1 / N \sum xcsq_i
    dxcsq = 1.0 / N * np.ones((N, D)) * dv
  
    # Step 3. xcsq = xc ^ 2
    dxc2 = 2.0 * dxcsq * xc
  
    # Step 2. xc = x - m
    dx1 = dxc1 + dxc2
    dm = - np.sum(dxc1 + dxc2, axis=0, keepdims=True)
  
    # Step 1. m = 1 / N \sum x_i
    dx2 = 1.0 / N * np.ones((N, D)) * dm
  
    dx = dx1 + dx2

    ###########################################################################
    #                             END OF CODE                                 #
    ###########################################################################

    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """Forward pass for inverted dropout.

    Note that this is different from the vanilla version of dropout.
    Here, p is the probability of keeping a neuron output, as opposed to
    the probability of dropping a neuron output.
    See http://cs231n.github.io/neural-networks-2/#reg for more details.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.
    """
    p, mode = dropout_param["p"], dropout_param["mode"]
    if "seed" in dropout_param:
        np.random.seed(dropout_param["seed"])

    mask = None
    out = None

    if mode == "train":
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # Generate random mask,
        # where the dimensions are *x.shape ('*' operator unpacks the tuple returned by the .shape into two attributes)
        # and comparing against p returns 1 if condition is matched or 0 otherwise
        # (note that dividing by p makes that you no longer need to update the prediction, this is the INVERTED DROPOUT) 

        mask = (np.random.rand(*x.shape) < p) / p

        # Generate the out by multipliyng (masking elements where the condition above got 0)
        out = x * mask

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # Nothing is performed in here, thanks to the inverted dropout !
        out = x 

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """Backward pass for inverted dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param["mode"]

    dx = None
    if mode == "train":
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # f(x) = mask * X => df / dx = mask 

        # So the product between the mask and dout is computed
        dx = dout * mask

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == "test":
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Get dimentions from inputs:
    N, _, H, W = x.shape
    F, _, HH, WW = w.shape
    pad, stride = conv_param['pad'], conv_param['stride']

    # Specify output dimentions
    out = np.zeros((N, F, int(1 + (H + 2 * pad - HH) / stride), int(1 + (W + 2 * pad - WW) / stride)))

    # Fill in the output

    for n in range(N): # For every image
        
        # Pad the nth image of shape [C, H, W]
        padded = np.pad(x[n, :], ((0, 0), (pad, pad), (pad, pad)), 'constant') # The (0,0) is for the Channels, the are not padded

        for f in range(F): # For every filter
          
            # Now we move through the image and convolve

            for i in range(0, H + 2 * pad - HH + 1, stride): # For every row - Note that the step ought to be stride!
                
                # Above we take into considerationn the padding and the size of the filter to avoid going out of bound!
                
                for j in range(0, W + 2 * pad - WW +1, stride): # For every column - Note that the step ought to be stride!
                    
                    # Above we take into considerationn the padding and the size of the filter to avoid going out of bound!
                    
                    # Convolve
                    out[n, f, i // stride, j // stride] = np.sum(padded[:, i:i + HH, j:j + WW] * w[f, :])

                    # Note that the floor division (//) is taken in order to get the indexes 
                    # at the output to be taken at step 1 instead of step stride !
                    # Otherwise the iteration should have been with step 1 and the stride taken in for loops.
                
            # After the convolution has been computed, the bias is added

            out[n,f,:,:] += b[f]
                        
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # Retrieve cached values
    x, w, b, conv_param = cache
    pad, stride = conv_param['pad'], conv_param['stride']
    
    # Get dimensions
    N, _, _, _ = x.shape
    F, _, HH, WW = w.shape
    _, _, H_out, W_out = dout.shape

    # Initialize gradients
    dx = np.zeros_like(x)
    dw = np.zeros_like(w)
    db = np.zeros_like(b)
    
    # out[n,f,i,j]=(convolution)+b[f]
    # => db = 1 * dout => SUMMATION

    # Compute db (gradient with respect to bias)
    db = np.sum(dout, axis=(0, 2, 3))
    # The above line sums over all the images (N), height (H_out), and width (W_out), leaving only the filter index F.

    # Pad the input x for dx computation
    x_padded = np.pad(x, ((0,0), (0,0), (pad,pad), (pad,pad)), mode='constant')
    dx_padded = np.pad(dx, ((0,0), (0,0), (pad,pad), (pad,pad)), mode='constant')

    # Compute dw and dx
    for n in range(N):  # Loop over each image
        
        for f in range(F):  # Loop over each filter
            
            for i in range(H_out):  # Loop over dout height
                
                for j in range(W_out):  # Loop over dout width
                    
                    # Find the slice in x_padded that contributed to this dout element:
                    # (multiplying by stride gives the specific indexes, given that on the forward we iterated over stride)
                    h_start = i * stride
                    h_end = h_start + HH
                    w_start = j * stride
                    w_end = w_start + WW

                    # Calculate dw (gradient with respect to the filter weights)
                    # out[n,f,i,j] = ∑∑∑ x[n,c,i+p,j+q]⋅w[f,c,p,q]+b[f]
                    # => dw[f,c,p,q] = ∑ dout[n,f,i,j]⋅x[n,c,i⋅stride+p,j⋅stride+q]
                    dw[f] += x_padded[n, :, h_start:h_end, w_start:w_end] * dout[n, f, i, j]
                    
                    # x_padded is the padded input,and the slice [h_start:h_end, w_start:w_end] corresponds to the portion of the 
                    # input that contributed to the current dout[n, f, i, j].

                    # Calculate dx (gradient with respect to the input)
                    # out[n,f,i,j] = ∑∑∑ x[n,c,i+p,j+q]⋅w[f,c,p,q]+b[f]
                    # dx[n,c,i,j]= ∑∑∑ dout[n,f,i−p,j−q]⋅w[f,c,p,q]

                    dx_padded[n, :, h_start:h_end, w_start:w_end] += w[f] * dout[n, f, i, j]

                    # dx_padded is updated by the convolution of the filters with the corresponding gradient dout[n, f, i, j]. 
                    # The region of dx affected by this gradient is updated accordingly.

    # Remove padding from dx_padded to get dx
    dx = dx_padded[:, :, pad:-pad, pad:-pad] if pad > 0 else dx_padded
                
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here, eg you can assume:
      - (H - pool_height) % stride == 0
      - (W - pool_width) % stride == 0

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max-pooling forward pass                            #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max-pooling backward pass                           #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization implemented above.               #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta