from sklearn.datasets import load_wine
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

wine = load_wine()
X, y = wine.data, wine.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

tree = DecisionTreeClassifier(max_depth=4, criterion='gini', random_state=42)
tree.fit(X_train, y_train)

y_pred = tree.predict(X_test)
print("정확도:", accuracy_score(y_test, y_pred))

print("혼동 행렬:\n", confusion_matrix(y_test, y_pred))

plt.figure(figsize=(12, 8))
plot_tree(tree, filled=True, feature_names=wine.feature_names, class_names=wine.target_names)
plt.title("Decision Tree for Wine Dataset (max_depth=4)")
plt.show()