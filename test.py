import numpy as np
import matplotlib.pyplot as plt

from core.NLP import preprocess, create_co_matrix, ppmi
from core.layers import MatMul


text = 'You say goodbye and I say hello.'

corpus, word_to_id, id_to_word = preprocess(text)

vocab_size = len(word_to_id)

C = create_co_matrix(corpus, vocab_size)

W = ppmi(C)

c = np.array([[1, 0, 0, 0, 0, 0, 0]])
W = np.random.randn(7, 3)
layer = MatMul(W)
h = layer.forward(c)
print(h)