from maths import numerical_gradient

def GD(f, init_x, lr=0.01, step_num=100):
  x = init_x

  for _ in range(step_num):
    grad = numerical_gradient(f, x)
    x -= lr * grad

  return x