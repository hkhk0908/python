from sklearn.datasets import load_iris
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

X,y = load_iris(return_X_y=True)

tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=0)
X_tsne = tsne.fit_transform(X)

plt.scatter(X_tsne[0:,0],X_tsne[:,1],c=y, cmap='viridis',edgecolor='k',s=50)
plt.title("t-SNE 시각화: Iris 데이터")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.grid(True)
plt.show()