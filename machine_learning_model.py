import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt

# Take input from the user for data points and labels
X = []
y = []
num_points = int(input("Enter the number of data points: "))
for i in range(num_points):
    point = input(f"Enter data point {i+1} as space-separated values: ").split()
    X.append([float(point[0]), float(point[1])])
    label = int(input(f"Enter label for data point {i+1} (1 or -1): "))
    y.append(label)

# Convert to NumPy arrays
X = np.array(X)
y = np.array(y)

# Train SVM
clf = svm.SVC(kernel='linear', C=1)
clf.fit(X, y)

# Extract model parameters
w = clf.coef_[0]
b = clf.intercept_[0]

print("Coefficients (w):", w)
print("Intercept (b):", b)

# Plot the data points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, marker='o', label='Data Points')

# Plot the decision boundary
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = clf.decision_function(xy).reshape(XX.shape)

# Plot decision boundary and margins
ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100, linewidth=1, facecolors='none', edgecolors='k', label='Support Vectors')

# Highlight the hyperplane
w_norm = np.linalg.norm(w)
slope = -w[0] / w[1]
intercept = -b / w[1]
x_hyperplane = np.linspace(xlim[0], xlim[1])
y_hyperplane = slope * x_hyperplane + intercept
plt.plot(x_hyperplane, y_hyperplane, linestyle='-', color='r', label='Hyperplane')

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('SVM Decision Boundary')
plt.legend()
plt.show()


# (1, 2) - label 1
# (2, 3) - label 1
# (3, 3) - label 1
# (4, 1) - label -1
# (5, 3) - label -1
# (6, 4) - label -1
