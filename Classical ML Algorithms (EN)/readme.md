# Fundamentals of **Machine Learning**
The aim of the project was to implement the most popular classical machine learning algorithms and use them on data sets. I decided to use a high-level language **C++** for this, including the **Eigen** library and the **CMake** tool, which is used to automatically manage the compilation process of multiple programs. The project was more challenging due to the lack of use of ready-made data processing libraries, such as **NumPy** or **Pandas** in **Python**.


## Data Loading
At the beginning, I implemented the **DataFrame** class, which has methods needed to load data from a **CSV** file, process data (such as dropping columns, rows, getting a slice of data called **slicing** or standardizing columns needed for training) and methods for visualization (such as head, displaying the first rows of data, or unique allowing to display the number of unique values in a given column). And functions needed to evaluate models, such as mean squared error (**MSE**) or Classification accuracy and a function to split data (DataFrame) into a training set and a validation set.


## **KMeans** Algorithm
This is the most popular **unsupervised learning** algorithm for **clustering** data. The algorithm initializes **centroids** with random points from the data, then iteratively assigns each point to the nearest centroid and updates the centroids based on the assigned points.

[**KMeans Implemenation**](https://github.com/JakubCzarnik/Machine-Learning-in-Cpp/tree/main/src/kmean)


## **k-Nearest Neighbors** (**KNN**) Algorithm
This is one of the simpler algorithms, as the fit method simply involves recording the dependent and target values. It classifies data based on its nearest neighbors in Euclidean space. To optimize calculations, I used the **heap** data structure.

[**KNN Implemenation**](https://github.com/JakubCzarnik/Machine-Learning-in-Cpp/tree/main/src/knn)


## **Linear Regression** and **Logistic Regression** Algorithms
Both of these algorithms are very similar, the difference is that logistic regression transfers the range of regression results to a discrete space (0, 1) so that it can classify the problem binary, using a **threshold**. For training, I used the MSE loss function, iterative gradient calculation, and the Stochastic Gradient Descent (**SGD**) algorithm.

[**Linear Regression Implemenation**](https://github.com/JakubCzarnik/Machine-Learning-in-Cpp/tree/main/src/linear_reggresion)

[**Logistic Regression Implemenation**](https://github.com/JakubCzarnik/Machine-Learning-in-Cpp/tree/main/src/logistic_reggresion)

## **Naive Bayes** Algorithm
It uses **Bayes’ theorem** for data classification. The fit method is responsible for training the model, it calculates the mean, variance, and a priori probability for each class. The predict method is responsible for predicting classes for new data, it calculates probabilities for each class, and then selects the class with the highest probability for each example.

[**Naive Bayes Implemenation**](https://github.com/JakubCzarnik/Machine-Learning-in-Cpp/tree/main/src/naive)

## **Principal Component Analysis** (**PCA**) Algorithm
The PCA algorithm is a dimensionality reduction technique, which is often used to visualize high-dimensional data and to speed up machine learning algorithms. The fit method is responsible for training the model, it centers the data, calculates the covariance matrix, and then calculates the **eigenvalues** and **eigenvectors** of the covariance matrix. The transform method is responsible for transforming data into a space with fewer dimensions.

[**PCA Implemenation**](https://github.com/JakubCzarnik/Machine-Learning-in-Cpp/tree/main/src/pca)

## Github Repository
The implementation of the entire project, along with its descriptions, can be found [**here**](https://github.com/JakubCzarnik/Machine-Learning-in-Cpp).