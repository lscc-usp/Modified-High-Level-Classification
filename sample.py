# Main structure of the code adapated from:
# Code source: Gael Varoquaux
#              Andreas Muller
# Modified for documentation by Jaques Grobler
# License: BSD 3 clause

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import DecisionBoundaryDisplay
from high_level import HighLevelClassification
import math

is_show_acc = True # Show obtained accuracy value in figure

# Creates a Circle of Data Points
def circle2D(r,n=20):
    return np.array([(math.cos(2*math.pi/n*x)*r,math.sin(2*math.pi/n*x)*r) for x in range(0,n)])

# Creates a Triangle of Data Points
def triangle2D(base,dist):
    x0 = 0
    base = int(base)
    total = (base*(base+1))/2
    points = np.zeros((int(total),2))
    height = base
    actual = 0

    for h in range(height):
        for x in range(base):
            points[actual,0] = x0+x*dist
            points[actual,1] = h*dist
            actual += 1
        x0 = x0+(dist/2)
        base -= 1

    return points

# Sample Experiment
# Creating input data
circle = circle2D(0.5, 10)
circle[:,0] += 1.5
circle[:,1] += 3.3
y_circle = np.zeros(len(circle))

tri = triangle2D(4,1)
y_tri = np.zeros(len(tri))+1

X = np.concatenate([circle, tri])
y = np.concatenate([y_circle, y_tri])

circle_triangle = (X, y)

#plt.scatter(X[:, 0], X[:, 1]) # for DEBUG
#plt.show()
#print('wait')

names = [
    "AdaBoost",
    "Decision Tree",
    "Logistic Regression",
    "MLP",
    "Naive Bayes",
    "Random Forest",
    "RBF SVM",
    "Deep MLP",
    "High-Level Network-Based",
]

classifiers = [
    AdaBoostClassifier(),
    DecisionTreeClassifier(max_depth=5),
    LogisticRegression(),
    MLPClassifier(),
    GaussianNB(),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    SVC(gamma=2, C=1),
    MLPClassifier(hidden_layer_sizes=(256, 128, 64, 32), activation="relu", random_state=1),
    HighLevelClassification(k=2, num_class=2, is_weighted = False, is_debug= True),
]

datasets = [
    circle_triangle,
    circle_triangle, # Distinct behavior (hard-coded inside iteration)
]

figure = plt.figure(figsize=(27, 9))
i = 1
# Iterate over datasets
for ds_cnt, ds in enumerate(datasets):
    # preprocess dataset, split into training and test part
    X, y = ds
    X = StandardScaler().fit_transform(X)

    if ds_cnt == 0 and ds == circle_triangle:
        X_train, y_train = X[0:len(X)-1], y[0:len(y)-1]
        X_test = X[len(X)-1].reshape((1, 2))
        y_test = np.array([1])
    elif ds_cnt == 1 and ds == circle_triangle:
        X_train, y_train = X, y
        X_test = X[len(X)-1].reshape((1, 2))
        y_test = np.array([1])
    else: # For usual train/test split of data sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.4, random_state=78
        )

    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    # just plot the dataset first
    cm = ListedColormap(["#007FFF", "#007F3F"]) # For Decision Boundary
    cm_bright = ListedColormap(["#007FFF", "#FF0000", "#007F3F"]) # Blue (Circle Class), Green (Triangle Class), Red (Test Sample)
    ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
    if ds_cnt == 0:
        ax.set_title("Input training data")
    # Plot the training points
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)#, edgecolors="k")
    # Plot the testing points
    if (ds_cnt == 0 or ds_cnt == 1) and ds == circle_triangle: # Testing points are not part of the input for training
        pass                                                   # specific for this scenario
    else:
        ax.scatter(
            X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6, edgecolors="k"
        )
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(())
    ax.set_yticks(())
    i += 1

    # iterate over classifiers
    for name, clf in zip(names, classifiers):
        ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
        i += 1
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)

        if type(clf) is HighLevelClassification:
            clf.plotNet(X_train, y_train, X_test = X_test, ax = ax, cmap = cm_bright, mode = 'test') # Network plot
        else:
            DecisionBoundaryDisplay.from_estimator(
                clf, X, cmap=cm, alpha=0.4, ax=ax, eps=0.5
            )
            # Plot the training points
            ax.scatter(
                X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright
            )
            # Plot the testing points
            ax.scatter(
                X_test[:, 0],
                X_test[:, 1],
                c='r',
                marker='o',
                cmap=cm_bright,
                edgecolors="r",
            )

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xticks(())
        ax.set_yticks(())
        if ds_cnt == 0:
            ax.set_title(name)
        if is_show_acc:
            ax.text(
                x_max - 0.3,
                y_min + 0.3,
                ("%.2f" % score).lstrip("0"),
                size=15,
                horizontalalignment="right",
            )        

plt.tight_layout()
#plt.savefig('fig1.png', dpi=300) # for DEBUG
plt.show(block=False)
plt.pause(0.01)


# Single Modified High-Level Classifier Call (for k=5)
X, y = circle_triangle
X = StandardScaler().fit_transform(X)
X_train = X[0:len(X)-1]
y_train = y[0:len(y)-1]
X_test = X[len(X)-1]
X_test = X_test.reshape((1, 2))
y_test = [1]

HC = HighLevelClassification(k=5, num_class=2, is_weighted = False, is_debug= True)
HC.fit(X_train, y_train)

# Network plot
HC.plotNet(X_train, y_train)
HC.plotNet(X_train, y_train, X_test, mode = 'test')

predicted_labels = HC.predict(X_test)
acc = HC.accuracy_score(predicted_labels, y_test)
print(acc)

print('Done! Waiting for figures to be closed...')
plt.show(block=True) # Deals with block = False (otherwise figures become unresponsive)
input('Press any key to finish...')

    



