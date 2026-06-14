import numpy as np

class Sigmoid():
  def __init__(self):
    self.out = None

  def forward(self, x):
    out = 1 / (1 + np.exp(-x))
    self.out = out
    return out
  
  def backward(self, dt):
    dx = dt * (1.0 - self.out) * self.out
    return dx

class Relu:
  def __init__(self):
    self.mask = None

  def forward(self, x):
    self.mask = (x <=0)
    out = x.copy()
    out[self.mask] = 0
    return out

  def backward(self, dt):
    dt[self.mask] = 0
    dx = dt
    return dx

def softmax(a):
  c = np.max(a)
  exp_a = np.exp(a - c)
  sum_exp_a = np.sum(exp_a)
  y = exp_a / sum_exp_a

  return y