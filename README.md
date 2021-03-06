# Digit-Recognizer

The data set can be obtained from https://www.kaggle.com/oddrationale/mnist-in-csv

This Repo contains the steps carried out towards  the design and  implementation of the Nearest Neighbour (NN) classifier from scratch using just numpy, to recognize the handwritten digits. The designed classifier was evaluated using the MNIST dataset and compared with Support Vector Machine (SVM), Random Forest and the Python in-built K-NN classifier available in the Sklearn library.

The comaparative results are show below.

In order to excute the code simply copy the data in the current working directory of Python and run section by section.

###### *Note: Seed was not set, as a result, the predictive results may slightly vary everytime the code is excuted

# Results

## Comparision based on Classification Accuracy and Computation Time :

![](Results/ComparisionT.PNG)

## Confusion Matrix of the models :
![](Results/NNcm.png)

![](Results/KNNcm.png)

![](Results/SVMcm.png)

![](Results/RFcm.png)





