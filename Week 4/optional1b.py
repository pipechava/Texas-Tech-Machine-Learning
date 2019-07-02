#Optional 1b

import pickle
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics

# Load the training and test data from the Pickle file
with open("mnist_dataset_unscaled.pickle", "rb") as f:
      train_data, train_labels, test_data, test_labels = pickle.load(f)

# Scale the training and test data using a standard distribution
pixel_mean = np.mean(train_data)
pixel_std = np.std(train_data)
train_data = (train_data - pixel_mean) / pixel_std
test_data = (test_data - pixel_mean) / pixel_std

# Count the unique classes
class_list = np.unique(train_labels)
num_classes = len(class_list)
num_pixels = train_data.shape[1]
num_test_cases = len(test_labels)

# Compute the centroid image
centroid_img = np.mean(train_data, 0)

# Loop through every class
prob_class_img = np.zeros( (num_classes, num_pixels) )
for classidx in range(num_classes):
  
  # Create an image of average pixels for this class
  mask = train_labels==classidx
  train_data_this_class = np.compress(mask, train_data, axis=0)
  centroid_img_class = np.mean(train_data_this_class, 0)

  # Compute probability of class for each pixel
  prob_class_img[classidx] = centroid_img_class / (centroid_img+1)

# Now use the probability images to estimate the probability of each class
# in new images
norm = plt.Normalize(0, 1024)
Y_pred = np.zeros(num_test_cases)

# Predict all test images
for testidx in range(num_test_cases):

  test_img = test_data[testidx]

  prob_class = []
  for classidx in range(num_classes):
    test_img_prob_class = test_img * prob_class_img[classidx]
    # Average the probabilities of all pixels
    prob_class.append( np.mean(test_img_prob_class) )

  # Pick the largest
  Y_pred[testidx] = prob_class.index(max(prob_class))

# Confusion matrix
cmatrix = sklearn.metrics.confusion_matrix(test_labels, Y_pred)
print("Confusion Matrix:")
print(cmatrix)

# Accuracy, precision & recall
print("Accuracy:   {:.3f}".format(sklearn.metrics.accuracy_score(test_labels, Y_pred)))
print("Precision:  {:.3f}".format(sklearn.metrics.precision_score(test_labels, Y_pred, average='weighted')))
print("Recall:     {:.3f}".format(sklearn.metrics.recall_score(test_labels, Y_pred, average='weighted')))

# Per-Class Precision & Recall
precision = sklearn.metrics.precision_score(test_labels, Y_pred, average=None)
recall = sklearn.metrics.recall_score(test_labels, Y_pred, average=None)
for n in range(num_classes):
    print("  Class {}: Precision: {:.3f} Recall: {:.3f}".format(n, precision[n], recall[n]))
    
'''
imbalaced classes:
    Accuracy:   0.622
    Precision:  0.779
    Recall:     0.622
    
balanced clases:
    Accuracy:   0.738
    Precision:  0.800
    Recall:     0.738
'''
