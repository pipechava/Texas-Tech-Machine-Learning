# Practice 2

import pickle
import numpy as np
import matplotlib.pyplot as plt
import sklearn.preprocessing
import sklearn.linear_model

# Load the training and test data from the Pickle file
with open("mnist_dataset_unscaled.pickle", "rb") as f:
      train_data, train_labels, test_data, test_labels = pickle.load(f)

# Scale the training and test data
pixel_mean = np.mean(train_data)
pixel_std = np.std(train_data)
train_data = (train_data - pixel_mean) / pixel_std
test_data = (test_data - pixel_mean) / pixel_std

# One-hot encode the labels
encoder = sklearn.preprocessing.OneHotEncoder(sparse=False, categories='auto')
train_labels_onehot = encoder.fit_transform(train_labels.reshape(-1, 1))
test_labels_onehot = encoder.transform(test_labels.reshape(-1, 1))
num_classes = len(encoder.categories_[0])

# Train a linear regression classifier
model = sklearn.linear_model.LinearRegression(fit_intercept=False)
model.fit(train_data, train_labels_onehot)

# Predict the probabilities of each class
Y_pred_proba = model.predict(test_data)

# Pick the maximum
Y_pred = np.argmax(Y_pred_proba, axis=1).astype("uint8")

# Accuracy, precision & recall
print("Accuracy:   {:.3f}".format(sklearn.metrics.accuracy_score(test_labels, Y_pred)))
print("Precision:  {:.3f}".format(sklearn.metrics.precision_score(test_labels, Y_pred, average='weighted')))
print("Recall:     {:.3f}".format(sklearn.metrics.recall_score(test_labels, Y_pred, average='weighted')))

# cycle through examples and get 10 wrong ones:
count = 0
for x in range(len(test_data)):
    if Y_pred[x] != test_labels[x]:
        image = test_data[x].reshape(28,28)
        plt.figure()
        plt.imshow(image, cmap="gray_r")
        plt.title("Predicted: "+str(Y_pred[x])+" Correct: "+str(test_labels[x]))
        plt.show()
        count = count +1
    if count == 10:
        break

