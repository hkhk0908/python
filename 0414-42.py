from sklearn.decomposition import FastICA
from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
n_samples = 2000
time = np.linspace(0, 8, n_samples)
s1 = np.sin(2 * time)
s2 = np.sign(np.sin(3 * time))
S = np.c_[s1, s2] + 0.1 * np.random.normal(size=(n_samples, 2))

A = np.array([[1, 1], [0.5, 2]])
X = S @ A.T

ica = FastICA(n_components=2)
S_ = ica.fit_transform(X)

plt.figure(figsize=(10,6))
titles = ['혼합된 신호', 'ICA 분리된 신호', '원본 신호']
for i, sig in enumerate([X, S_, S]):
    plt.subplot(3,1,i+1)
    plt.plot(sig)
    plt.title(titles[i])
plt.tight_layout()
plt.show()