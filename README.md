# Predicting-Student-Performance-in-Mathematics
The objective of this project is to build and compare two binary classifiers for predicting student performance in Mathematics, using the data collected from two public schools in Portugal during the school year 2005/06.
# Importing libraries
import pandas as pd
import numpy as np
# Loading dataset 
df_m= pd.read_csv("student-mat.csv", sep=";")
print(df_m.shape)
df_m.head()
df_m.loc[(df_m .G3< 10), ('G3')] = 0
df_m.loc[(df_m .G3>= 10), ('G3')] =1
df_m
print(df_m.shape)
df_m.columns.values
The dataset contains 395 observations. The numeric target feature "G3" is the  "target" and transformed into a binary categorical feature with two levels "1"="pass" and "0"="fail".


df_m
# Checking for Missing Values
df_m.isna().sum()
# Statistical summary
df_m.describe(include='all')
# Partitioning
dat = df_m.drop(columns='G3')
target = df_m['G3']
dat
dat1=dat.corr()
dat1
# Target Feature
It is obvious that the target classes are imbalanced. The number of "pass" is twice as many as that of "fail".
"1" means 'pass' and '0' means 'fail'
dat
target.value_counts()
# Categorical Descriptive Features
There are two types of categorical descriptive features in the dataset
# Nominal:
nominal_cols = dat.columns[dat.dtypes==object].tolist()
nominal_cols
for col in nominal_cols:
    n = len(dat[col].unique())
    if (n == 2):
        dat[col] = pd.get_dummies(dat[col], drop_first=True)
data = pd.get_dummies(dat)
data = pd.get_dummies(dat)
After performing one-hot-enconding on those 4 nominal features, the number of columns with descriptive features in the dataset extend from 32 to 45.
# Feature Selection
data1=data[['age','Medu','Fedu','traveltime','studytime','failures','famrel','freetime','goout','Dalc','Walc','health','absences','G1','G2']]
data1
# Feature Scaling
Scaling descriptive features is beneficial as it can normalise the numeric values among different variables within a specific range and can help speed up the processing time in the algorithm. Min-Max Scaling is applied to scale the descriptive features between 0 and 1. Each binary feature can be still kept as binary after scaling.
from sklearn import preprocessing
data_unscaled=data1.values
data_scaled = preprocessing.MinMaxScaler().fit_transform(data_unscaled)
pd.DataFrame(data_scaled, columns=data1.columns).head()
# Train-Test Splitting
The dataset is split into train and test at a 70:30 partition ratio by stratification:

Training (70%): X_train (descriptive), y_train (target)
Testing (30%): X_test (desciptive), y_test (target)
Meanwhile, I created X_train_10 and X_test_10, which have the same sample rows as X_train and X_test, but only have the top 10 features selected by F-score from the previous process.
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(data_scaled,
                                                 target.values,
                                                 test_size=0.3,
                                                 random_state=42,
                                                 stratify=target.values)
print(X_train.shape)
print(X_test.shape)
# 1. K-Nearest Neighbors
# For all features
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
def most_common(lst):
    return max(set(lst), key=lst.count)
def euclidean(point, data):
    # Euclidean distance between points a & data
    return np.sqrt(np.sum((point - data)**2, axis=1))
class KNeighborsClassifier:
    def __init__(self, k=5, dist_metric=euclidean):
        self.k = k
        self.dist_metric = dist_metric
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
    def predict(self, X_test):
        neighbors = []
        for x in X_test:
            distances = self.dist_metric(x, self.X_train)
            y_sorted = [y for _, y in sorted(zip(distances, self.y_train))]
            neighbors.append(y_sorted[:self.k])
        return list(map(most_common, neighbors))
    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        accuracy = sum(y_pred == y_test) / len(y_test)
        return accuracy

accuracies = []
ks = range(1, 30)
for k in ks:
    knn = KNeighborsClassifier(k=k)
    knn.fit(X_train, y_train)
    accuracy = knn.evaluate(X_test, y_test)
    accuracies.append(accuracy)
# Visualize accuracy vs. k
fig, ax = plt.subplots()
ax.plot(ks, accuracies)
ax.set(xlabel="k",
       ylabel="Accuracy",
       title="Performance of knn")
plt.show()
print(accuracies)
# Naive Bayes Classifier
1. Function to identify partition values for all attribute
X=data_scaled
X.shape
y=target.values
y.shape
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,
                                                 y,
                                                 test_size=0.3,
                                                 random_state=42,
                                                 stratify=target.values)
print(X_train.shape)
print(X_test.shape)
class NaiveBayes:

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)

        # calculate mean, var, and prior for each class
        self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self._var = np.zeros((n_classes, n_features), dtype=np.float64)
        self._priors = np.zeros(n_classes, dtype=np.float64)

        for idx, c in enumerate(self._classes):
            X_c = X[y == c]
            self._mean[idx, :] = X_c.mean(axis=0)
            self._var[idx, :] = X_c.var(axis=0)
            self._priors[idx] = X_c.shape[0] / float(n_samples)
            

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        posteriors = []

        # calculate posterior probability for each class
        for idx, c in enumerate(self._classes):
            prior = np.log(self._priors[idx])
            posterior = np.sum(np.log(self._pdf(idx, x)))
            posterior = posterior + prior
            posteriors.append(posterior)

        # return class with the highest posterior
        return self._classes[np.argmax(posteriors)]

    def _pdf(self, class_idx, x):
        mean = self._mean[class_idx]
        var = self._var[class_idx]
        numerator = np.exp(-((x - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator


# Testing
if __name__ == "__main__":
    # Imports
    from sklearn.model_selection import train_test_split
    from sklearn import datasets

    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy
   # X, y = datasets.make_classification(
    #    n_samples=1000, n_features=10, n_classes=2, random_state=123
    #)
    X_train,X_test,y_train,y_test = train_test_split(X,
                                                 y,
                                                 test_size=0.3,
                                                 random_state=42,
                                                 stratify=target.values)
   
    nb = NaiveBayes()
    nb.fit(X_train, y_train)
    predictions = nb.predict(X_test)#(X_test[0])
   # print(predictions)

    print("Naive Bayes classification accuracy", accuracy(y_test, predictions))
# For All features 
data
target
from sklearn import preprocessing
data_unscaled=data.values
data_scaled = preprocessing.MinMaxScaler().fit_transform(data_unscaled)
pd.DataFrame(data_scaled, columns=data.columns).head()
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(data_scaled,
                                                 target.values,
                                                 test_size=0.3,
                                                 random_state=42,
                                                 stratify=target.values)
print(X_train.shape)
print(X_test.shape)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
def most_common(lst):
    return max(set(lst), key=lst.count)
def euclidean(point, data):
    # Euclidean distance between points a & data
    return np.sqrt(np.sum((point - data)**2, axis=1))
class KNeighborsClassifier:
    def __init__(self, k=5, dist_metric=euclidean):
        self.k = k
        self.dist_metric = dist_metric
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
    def predict(self, X_test):
        neighbors = []
        for x in X_test:
            distances = self.dist_metric(x, self.X_train)
            y_sorted = [y for _, y in sorted(zip(distances, self.y_train))]
            neighbors.append(y_sorted[:self.k])
        return list(map(most_common, neighbors))
    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        accuracy = sum(y_pred == y_test) / len(y_test)
        return accuracy

accuracies = []
ks = range(1, 30)
for k in ks:
    knn = KNeighborsClassifier(k=k)
    knn.fit(X_train, y_train)
    accuracy = knn.evaluate(X_test, y_test)
    accuracies.append(accuracy)
# Visualize accuracy vs. k
fig, ax = plt.subplots()
ax.plot(ks, accuracies)
ax.set(xlabel="k",
       ylabel="Accuracy",
       title="Performance of knn")
plt.show()
print(accuracies)
