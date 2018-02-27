
# coding: utf-8

# # Texture synthesis and artistic style transfer
# 
# In this homework you are to imlement [A Neural algorithm of artistic style](http://arxiv.org/pdf/1508.06576v2.pdf). This is an extension of [Texture Synthesis Using Convolutional Neural Networks](http://arxiv.org/pdf/1505.07376v3.pdf) method.
# 
# The core of the method -- VGG and constrained optimization. The constrains are of two types: *content* and *style*. Given a content image **C** and style image **S** we want to generate an image **X** with content from **C** and style (whatever it really means) from **S**. 
# 
# We want to design a loss function for the optimization process. Considering \[1\], \[2\], an input image is easily invertable from the outputs at intermediate layers. This explains the idea of making an intermediate representation $F_X$ of **X** close to **C** representation $F_C$. 
# 
# $$
#    L_{content} = || F_X - F_C || \rightarrow \min_X
# $$
# 
# Note, that representation $F$ preserve spatial information. Idea: let us dismiss it, so we will know what objects are there on the picture, but will not be able to reestablish their localtion. The style can be thought as something independent of content, something we are left with if we let the content off. L. Gatys suggests to dismiss spatial information by computing correlations between the feature maps $F$. If $F$ has dimensions `CxWxH`, then correlation matrix will be `CxC`, and look there's no spatial dimentions. So the style term will be responsible for mathing these correlation (Gram) matrices. 
# 
# $$
#    L_{style} = || Gram(F_X) - Gram(F_C) || \rightarrow \min_X
# $$
# 
# And finaly we combine the two.
# 
# $$
#    L = \alpha L_{content} + \beta L_{style} \min_X
# $$
# 
# Read the paper and the code for the details on layers, features $F$ are got from.

# #### A little bit of history behind this texture generation method
# 
# Actually the idea comes from 90th, when mathematical models of texures were developed \[3\]. They defined a probabolistic model for texture generation. They used an idea, that two images are indeed two samples of a particular texture iff their statistics match. The statistics used are histograms of given texture $I$ filtered with a number of filters: $\{hist(F_i * I), \quad i = 1,\dots, k\}$. And whatever image has the same statistics is thought as a sample of texture $I$. The main drawback was the Gibbs sampling was employed (which is very slow). \[4\] suggested exactly the scheme we use now: starting from a random image, let's adjust its statistics iteratively so they match the desired. 
# 
# Now, what is changed: the filters. \[4\] used carefully crafted set of filters, and now we use neural network based non-linear filters. We still use the idea of matching statistics, but the statistics improved. 
# 
# \[1\] *A.Mahendran, A.Vedaldi [Understanding Deep Image Representations by Inverting Them](https://www.robots.ox.ac.uk/~vgg/publications/2015/Mahendran15/mahendran15.pdf)*
# 
# \[2\] *A.Dosovitsky, T.Brox [Inverting Visual Representations with Convolutional Networks](http://arxiv.org/pdf/1506.02753v3.pdf)*
# 
# \[3\] *Zhu et. al. Filters, 1997 [Random Fields and Maximum Entropy (FRAME):
# Towards a Unified Theory for Texture Modeling](http://www.stat.ucla.edu/~ywu/research/papers/ijcv.pdf)*
# 
# \[4\] *Portilla & Simoncelli, 2000  [A Parametric Texture Model Based on Joint Statistics
# of Complex Wavelet Coefficients](http://www.cns.nyu.edu/pub/lcv/portilla99-reprint.pdf)*

# # Homework

# Минимизировать расстояние между матрицами Грамма это значит минимизировать расстояние между дисперсиями и корреляциями случайных величин. 
# 
# $$Q(x) = N(0, \sigma)$$
# 
# $$KL(P||Q) = \int P(x) \log P(x) dx - \int P(x) \log Q(x) $$
# 
# $$\nabla_{\sigma} KL(P||Q) = \int P(x) \left( - \frac{x^2}{2 \sigma^3} + \frac{1}{\sigma} \right) = 0$$
# 
# $$\sigma^2 = \int P(x) x^2 dx = Var(P(x))$$
# 
# Т.к. совпадение матриц Грамма обеспечивает совпадение дисперсий, то получаем из этого уравнения, что KL-дивиргенция минимизируется при совпадении матриц Грамма.

# In[1]:


import lasagne
import numpy as np
import pickle
import skimage.transform
import scipy

import theano
import theano.tensor as T

from lasagne.utils import floatX

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from lasagne.layers import InputLayer, DenseLayer, NonlinearityLayer
from lasagne.layers import Conv2DLayer as ConvLayer
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.nonlinearities import softmax

IMAGE_W = 256

# Note: tweaked to use average pooling instead of maxpooling
def build_model():
    net = {}
    net['input'] = InputLayer((1, 3, IMAGE_W, IMAGE_W))
    net['conv1_1'] = ConvLayer(net['input'], 64, 3, pad=1)
    net['conv1_2'] = ConvLayer(net['conv1_1'], 64, 3, pad=1)
    net['pool1'] = PoolLayer(net['conv1_2'], 2, mode='average_exc_pad')
    net['conv2_1'] = ConvLayer(net['pool1'], 128, 3, pad=1)
    net['conv2_2'] = ConvLayer(net['conv2_1'], 128, 3, pad=1)
    net['pool2'] = PoolLayer(net['conv2_2'], 2, mode='average_exc_pad')
    net['conv3_1'] = ConvLayer(net['pool2'], 256, 3, pad=1)
    net['conv3_2'] = ConvLayer(net['conv3_1'], 256, 3, pad=1)
    net['conv3_3'] = ConvLayer(net['conv3_2'], 256, 3, pad=1)
    net['conv3_4'] = ConvLayer(net['conv3_3'], 256, 3, pad=1)
    net['pool3'] = PoolLayer(net['conv3_4'], 2, mode='average_exc_pad')
    net['conv4_1'] = ConvLayer(net['pool3'], 512, 3, pad=1)
    net['conv4_2'] = ConvLayer(net['conv4_1'], 512, 3, pad=1)
    net['conv4_3'] = ConvLayer(net['conv4_2'], 512, 3, pad=1)
    net['conv4_4'] = ConvLayer(net['conv4_3'], 512, 3, pad=1)
    net['pool4'] = PoolLayer(net['conv4_4'], 2, mode='average_exc_pad')
    net['conv5_1'] = ConvLayer(net['pool4'], 512, 3, pad=1)
    net['conv5_2'] = ConvLayer(net['conv5_1'], 512, 3, pad=1)
    net['conv5_3'] = ConvLayer(net['conv5_2'], 512, 3, pad=1)
    net['conv5_4'] = ConvLayer(net['conv5_3'], 512, 3, pad=1)
    net['pool5'] = PoolLayer(net['conv5_4'], 2, mode='average_exc_pad')

    return net


# In[2]:


net = build_model()

values = pickle.load(open('vgg19_normalized.pkl', mode='rb'), encoding='latin1')['param values']
lasagne.layers.set_all_param_values(net['pool5'], values)


# In[3]:


MEAN_VALUES = np.array([104, 117, 123]).reshape((3,1,1))

def prep_image(im):
    if len(im.shape) == 2:
        im = im[:, :, np.newaxis]
        im = np.repeat(im, 3, axis=2)
    h, w, _ = im.shape
    if h < w:
        im = skimage.transform.resize(im, (IMAGE_W, w*IMAGE_W//h), preserve_range=True)
    else:
        im = skimage.transform.resize(im, (h*IMAGE_W//w, IMAGE_W), preserve_range=True)

    # Central crop
    h, w, _ = im.shape
    im = im[h//2-IMAGE_W//2:h//2+IMAGE_W//2, w//2-IMAGE_W//2:w//2+IMAGE_W//2]
    
    rawim = np.copy(im).astype('uint8')
    
    # Shuffle axes to c01
    im = np.swapaxes(np.swapaxes(im, 1, 2), 0, 1)
    
    # Convert RGB to BGR
    im = im[::-1, :, :]

    im = im - MEAN_VALUES
    return rawim, floatX(im[np.newaxis])


# In[4]:


photo = plt.imread('./photos/15504091853_82aba2c440_o.jpg')
rawim, photo = prep_image(photo)
plt.imshow(rawim)


# In[5]:


art = plt.imread('./style1.jpg')
rawim, art = prep_image(art)
plt.imshow(rawim)


# In[6]:


def gram_matrix(x):
    x = x.flatten(ndim=3)
    g = T.tensordot(x, x, axes=([2], [2]))
    return g


def content_loss(P, X, layer):
    p = P[layer]
    x = X[layer]
    
    loss = 1./2 * ((x - p)**2).sum()
    return loss


def style_loss(A, X, layer):
    a = A[layer]
    x = X[layer]
    
    A = gram_matrix(a)
    G = gram_matrix(x)
    
    N = a.shape[1]
    M = a.shape[2] * a.shape[3]
    
    loss = 1./(4 * N**2 * M**2) * ((G - A)**2).sum()
    return loss

def total_variation_loss(x):
    return (((x[:,:,:-1,:-1] - x[:,:,1:,:-1])**2 + (x[:,:,:-1,:-1] - x[:,:,:-1,1:])**2)**1.25).sum()


# In[ ]:


layers = ['conv4_2', 'conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
layers = {k: net[k] for k in layers}

input_im_theano = T.tensor4()
outputs = lasagne.layers.get_output(layers.values(), input_im_theano)

photo_features = {k: theano.shared(output.eval({input_im_theano: photo}))
                  for k, output in zip(layers.keys(), outputs)}
art_features = {k: theano.shared(output.eval({input_im_theano: art}))
                for k, output in zip(layers.keys(), outputs)}

generated_image = theano.shared(floatX(np.random.uniform(-128, 128, (1, 3, IMAGE_W, IMAGE_W))))

gen_features = lasagne.layers.get_output(layers.values(), generated_image)
gen_features = {k: v for k, v in zip(layers.keys(), gen_features)}


# In[ ]:


# Define loss function
losses = []


# style loss
losses.append(0.2e6 * style_loss(art_features, gen_features, 'conv1_1'))
losses.append(0.2e6 * style_loss(art_features, gen_features, 'conv2_1'))
losses.append(0.2e6 * style_loss(art_features, gen_features, 'conv3_1'))
losses.append(0.2e6 * style_loss(art_features, gen_features, 'conv4_1'))
losses.append(0.2e6 * style_loss(art_features, gen_features, 'conv5_1'))

# total variation penalty
losses.append(0.1e-7 * total_variation_loss(generated_image))

total_loss = sum(losses)

grad = T.grad(total_loss, generated_image)

# Theano functions to evaluate loss and gradient
f_loss = theano.function([], total_loss)
f_grad = theano.function([], grad)


# In[ ]:


# Helper functions to interface with scipy.optimize
def eval_loss(x0):
    x0 = floatX(x0.reshape((1, 3, IMAGE_W, IMAGE_W)))
    generated_image.set_value(x0)
    return f_loss().astype('float64')

def eval_grad(x0):
    x0 = floatX(x0.reshape((1, 3, IMAGE_W, IMAGE_W)))
    generated_image.set_value(x0)
    return np.array(f_grad()).flatten().astype('float64')

# Initialize with a noise image
generated_image.set_value(floatX(np.random.uniform(-128, 128, (1, 3, IMAGE_W, IMAGE_W))))

x0 = generated_image.get_value().astype('float64')
x_style = []
x_style.append(x0)

# Optimize, saving the result periodically
for i in range(2):
    print(i)
    scipy.optimize.fmin_l_bfgs_b(eval_loss, x0.flatten(), fprime=eval_grad, maxfun=40)
    x0 = generated_image.get_value().astype('float64')
    x_style.append(x0)


# In[ ]:


def deprocess(x):
    x = np.copy(x[0])
    x += MEAN_VALUES

    x = x[::-1]
    x = np.swapaxes(np.swapaxes(x, 0, 1), 1, 2)
    
    x = np.clip(x, 0, 255).astype('uint8')
    return x

plt.figure(figsize=(12,12))
for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.gca().xaxis.set_visible(False)    
    plt.gca().yaxis.set_visible(False)    
    plt.imshow(deprocess(x_style[i]))
plt.tight_layout()


# In[ ]:


# Define loss function
losses = []


# style loss
losses.append(0.2e6 * style_loss(art_features, gen_features, 'conv1_1'))
losses.append(0.2e6 * style_loss(art_features, gen_features, 'conv2_1'))
losses.append(0.2e6 * style_loss(art_features, gen_features, 'conv3_1'))
losses.append(0.2e6 * style_loss(art_features, gen_features, 'conv4_1'))
losses.append(0.2e6 * style_loss(art_features, gen_features, 'conv5_1'))

# total variation penalty
losses.append(0.1e-7 * total_variation_loss(generated_image))

# content loss
losses.append(0.03 * content_loss(photo_features, gen_features, 'conv4_2'))

total_loss = sum(losses)

grad = T.grad(total_loss, generated_image)

# Theano functions to evaluate loss and gradient
f_loss = theano.function([], total_loss)
f_grad = theano.function([], grad)

# Initialize with a noise image
generated_image.set_value(floatX(np.random.uniform(-128, 128, (1, 3, IMAGE_W, IMAGE_W))))

x0 = generated_image.get_value().astype('float64')
x_content = []
x_content.append(x0)

# Optimize, saving the result periodically
for i in range(2):
    print(i)
    scipy.optimize.fmin_l_bfgs_b(eval_loss, x0.flatten(), fprime=eval_grad, maxfun=40)
    x0 = generated_image.get_value().astype('float64')
    x_content.append(x0)


# In[ ]:


plt.figure(figsize=(12,12))
for i in range(9):
    plt.subplot(2, 1, i+1)
    plt.gca().xaxis.set_visible(False)    
    plt.gca().yaxis.set_visible(False)    
    plt.imshow(deprocess(x_content[i]))
plt.tight_layout()

