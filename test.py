import numpy as np
import matplotlib.pyplot as plt

from proto.NLP import preprocess, create_co_matrix, ppmi


text = 'You say goodbye and I say hello.'

corpus, word_to_id, id_to_word = preprocess(text)

vocab_size = len(word_to_id)

C = create_co_matrix(corpus, vocab_size)

W = ppmi(C)

U, S, V = np.linalg.svd(W)