# Load libraries
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Loading in the dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names_of_columns = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names_of_columns)

# # Looking at the dimensions or shape of the dataset
# print(dataset.shape)
#
# # Looking at the data itself,
# print(dataset.head(20))
#
# # Statistical summary of the data
# print(dataset.describe())
#
# # Breaking down the data by the class variable, seeing how many of each unique class
# print(dataset.groupby('class').size())


# Univariate plot, plots of each individual variable
# Since inputs are numeric, box makes the most sense
dataset.plot(
    kind='box',
    subplots=True,
    layout=(2, 2),
    sharex=False,
    sharey=False
)

#pyplot.show()

# Histogram
dataset.hist()
#pyplot.show()

# Multivariable plot, plots to see the interaction between the variables
# Starting with scatter to see structure and the relationships between the inputs

scatter_matrix(dataset)
#pyplot.show()

# Creating a validation dataset to have a set that is independent of training for comparison purposes

# Splitting of validation dataset
array = dataset.values
X = array[:, 0:4]
y = array[:, 4]

X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.2, random_state=1)


# Testing of 6 different types of algorithms:
# 1. Logistic Regression (LR)
# 2. Linear Discriminant Analysis (LDA)
# 3. K-Nearest Neighbors (KNN).
# 4. Classification and Regression Trees (CART).
# 5. Gaussian Naive Bayes (NB).
# 6. Support Vector Machines (SVM).
# Simple Linear are "Linear Regression" and "Linear Discriminant Analysis"
# Nonlinear are "K-Nearest Neighbors","Classification and Regression Trees",
#               "Gaussian Naive Bayes", and "Support Vector Machines"

# Spot Check Algorithms
models = []
models.append(('LR',
               LogisticRegression(
                   solver='liblinear',
                   multi_class='ovr')
               ))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))

# evaluating each model in turn
results = []
names = []
for name, model in models:
    k_fold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, X_train, Y_train, cv=k_fold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s: \n\tMean: %f \n\tSTD: %f\n' % (name, cv_results.mean(), cv_results.std()))

# Compare Algorithms
pyplot.boxplot(results, labels=names)
pyplot.title('Algorithm Comparison')
# pyplot.show()

# Making predictions of validation dataset
model = SVC(gamma='auto')
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)

# Evaluate predictions
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

