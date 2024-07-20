import numpy as np

from activations import softmax
from losses import cross_entropy_error

class Linear:
  def __init__(self, W, b):
    self.W = W
    self.b = b
    self.x = None
    self.dW = None
    self.db = None

  def forward(self, x):
    self.x = x
    out = np.dot(x, self.W) + self.b

    return out
  
  def backward(self, dt):
    dx = np.dot(dt, self.W.T)
    self.dW = np.dot(self.x.T, dt)
    self.db = np.sum(dt, axis=0)

    return dx
  
class SoftmaxWithLoss:
  def __init__(self):
    self.loss = None
    self.y = None
    self.t = None

  def forward(self, x, t):
    self.t = t
    self.y = softmax(x)
    self.loss = cross_entropy_error(self.y, self.t)
    return self.loss
  
  def backward(self, dt=1):
    batch_size = self.t.shape[0]
    dx = (self.y - self.t) / batch_size

    return dx