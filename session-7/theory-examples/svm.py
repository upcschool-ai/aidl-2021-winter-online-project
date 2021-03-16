from sklearn import datasets, svm
from sklearn.metrics import accuracy_score

# Load the iris dataset
iris_X, iris_y = datasets.load_iris(return_X_y=True)

# Split the data into training/testing sets
iris_X_train = iris_X[:-20]
iris_X_test = iris_X[-20:]

# Split the targets into training/testing sets
iris_y_train = iris_y[:-20]
iris_y_test = iris_y[-20:]

# Create Support Vector Classifier object
clf = svm.SVC()

# Train the model using the training sets
clf.fit(iris_X_train, iris_y_train)

# Make predictions using the testing set
iris_y_pred = clf.predict(iris_X_test)

# Accuracy
print(f'Accuracy: {accuracy_score(iris_y_test, iris_y_pred):.2f}')
# 90% accuracy
