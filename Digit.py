import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import time
import pandas as pd

#%%
def load_data(data_dir):
    train_data = open(data_dir + "mnist_train.csv").read()
    train_data = train_data.split("\n")[1:-1]
    train_data = [i.split(",") for i in train_data]
    X_train = np.array([[int(i[j]) for j in range(1,len(i))] for i in train_data])
    y_train = np.array([int(i[0]) for i in train_data])

    test_data = open(data_dir + "mnist_test.csv").read()
    test_data = test_data.split("\n")[1:-1]
    test_data = [i.split(",") for i in test_data]
    X_test = np.array([[int(i[j]) for j in range(1,len(i))] for i in test_data])
    y_test = np.array([int(i[0]) for i in test_data])

    return X_train, y_train, X_test, y_test
#%%

# A KNN Classifier coded from scratch

class simple_knn():

    def __init__(self):
        pass

    def train(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X, k=1):
        dists = self.compute_distances(X)
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)

        for i in range(num_test):
            k_closest_y = []
            
            labels = self.y_train[np.argsort(dists[i,:])].flatten()
            k_closest_y = labels[:k]
            c = Counter(k_closest_y)
            y_pred[i] = c.most_common(1)[0][0]
        return(y_pred)

    def compute_distances(self, X):
        dot_pro = np.dot(X, self.X_train.T)
        sum_square_test = np.square(X).sum(axis = 1)
        sum_square_train = np.square(self.X_train).sum(axis = 1)
        dists = np.sqrt(-2 * dot_pro + sum_square_train + np.matrix(sum_square_test).T)
        return(dists)
  
#%%
data_dir = "C:/Users/cool_/"
X_train, y_train, X_test, y_test = load_data(data_dir)
#%%

classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
batch_size = 1000
k = 1
classifier = simple_knn()
classifier.train(X_train, y_train)
predictions = []

for i in range(int((len(X_test)+1)/(batch_size))):
    print("Computing batch " + str(i+1) + "/" + str(int((len(X_test)+1)/batch_size)) + "...")
    tic = time.time()
    predts = classifier.predict(X_test[i * batch_size:(i+1) * batch_size], k)
    toc = time.time()
    predictions = predictions + list(predts)
    print("Completed this batch in " + str(toc-tic) + " Secs.")

y_pred = (np.array(predictions))
y_pred = y_pred.astype(int)

from sklearn.metrics import confusion_matrix,accuracy_score

cm = confusion_matrix(y_test, y_pred)

import seaborn as sns

cm_plot = sns.heatmap(cm, annot=True ,cmap="YlGnBu",linewidths=.1,fmt="d" )

plt.title("Confusion Matrix Nearest Neighbour(Scratch)")
plt.xlabel("Predicted Digit Label")
plt.ylabel("Actual Digit Label")
plt.show()

Accuracy = accuracy_score(y_test, y_pred)


#%%RF

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_jobs=-1, n_estimators=10)
rfc.fit(X_train, y_train)

rfc.score(X_test, y_test)
y_out = rfc.predict(X_test)
cmrf = confusion_matrix(y_test, y_out)

Accuracyrf = accuracy_score(y_test, y_out)

cm_plot = sns.heatmap(cmrf, annot=True ,cmap="YlGnBu",linewidths=.1,fmt="d" )

plt.title("Confusion Matrix Random-Forest Decision Trees")
plt.xlabel("Predicted Digit Label")
plt.ylabel("Actual Digit Label")
plt.show()


#%%SVM

from sklearn.svm import SVC
print('SVM Classifier with gamma = 0.1; Kernel = Polynomial')
classifier = SVC(gamma=0.1, kernel='poly', random_state = 0)
classifier.fit(X_train,y_train)

y_predsvm = classifier.predict(X_test)
from sklearn.metrics import accuracy_score,confusion_matrix
model_acsvmc = classifier.score(X_test, y_test)
test_accsvm = accuracy_score(y_test, y_predsvm)
conf_matsvm = confusion_matrix(y_test,y_predsvm)

print('\nSVM Trained Classifier Accuracy: ', model_acsvmc)
print('\nPredicted Values: ',y_predsvm)
print('\nAccuracy of Classifier on Validation Images: ',test_accsvm)

cm_plot = sns.heatmap(conf_matsvm, annot=True ,cmap="YlGnBu",linewidths=.1,fmt="d" )

plt.title("Confusion Matrix SVM")
plt.xlabel("Predicted Digit Label")
plt.ylabel("Actual Digit Label")
plt.show()


#%%KNN

from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=1)
model.fit(X_train, y_train)
predictionsknn = model.predict(X_test)

cmknn = confusion_matrix(y_test,predictionsknn)
test_accknn = accuracy_score(y_test, predictionsknn)

cm_plot = sns.heatmap(cmknn, annot=True ,cmap="YlGnBu",linewidths=.1,fmt="d" )

plt.title("Confusion Matrix KNN")
plt.xlabel("Predicted Digit Label")
plt.ylabel("Actual Digit Label")
plt.show()

