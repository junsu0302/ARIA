import numpy as np
from collections import OrderedDict

from proto.activations import sigmoid, softmax, Relu
from proto.losses import cross_entropy_error
from proto.maths import numerical_gradient
from proto.layers import Linear, SoftmaxWithLoss

class TwoLayerNet:
  def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
    self.params = {}
    self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
    self.params['b1'] = np.zeros(hidden_size)
    self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
    self.params['b2'] = np.zeros(hidden_size)

    self.layers = OrderedDict()
    self.layers['Linear1'] = Linear(self.params['W1'], self.params['b1'])
    self.layers['Relu1'] = Relu()
    self.layers['Linear2'] = Linear(self.params['W2'], self.params['b2'])
    
    self.lastLayer = SoftmaxWithLoss()

  def predict(self, x):
    for layer in self.layers.values():
      x = layer.forward(x)

    return x
  
  def loss(self, x, t):
    y = self.predict(x)

    return self.lastLayer.forward(y, t)
  
  def accuracy(self, x, t):
    y = self.predict(x)
    y = np.argmax(y, axis=1)
    if t.ndim != 1: 
      t = np.argmax(t, axis=1)

    accuracy = np.sum(y == t) / float(x.shape[0])
    return accuracy
  
  def numerical_gradient(self, x, t):
    loss_W = lambda W: self.loss(x, t)

    grads = {}
    grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
    grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
    grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
    grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

    return grads
  
  def gradient(self, x, t):
    self.loss(x, t)

    dt = 1
    dt = self.lastLayer.backward(dt)

    layers = list(self.layers.values())
    layers.reverse()
    for layer in layers:
      dt = layer.backward(dt)

    grads = {}
    grads['W1'] = self.layers['Linear1'].dW
    grads['b1'] = self.layers['Linear1'].db
    grads['W2'] = self.layers['Linear2'].dW
    grads['b2'] = self.layers['Linear2'].db

    return grads