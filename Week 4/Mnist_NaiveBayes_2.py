# Introduction to Artificial Intelligence
# Naive Bayes classifier, Part 2
# By Juan Carlos Rojas

import pickle
import numpy as np
import matplotlib.pyplot as plt

# Load the training and test data from the Pickle file
with open("mnist_dataset_unscaled.pickle", "rb") as f:
      train_data, train_labels, test_data, test_labels = pickle.load(f)

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

# Predict a few test images
for testidx in range(5):

  test_img = test_data[testidx]

  # Draw the test image
  plt.figure()
  plt.imshow(test_img.reshape(28,28), cmap="gray_r")
  plt.title("Test Image")
  plt.show()

  prob_class = []
  plt.figure()
  for classidx in range(num_classes):

    test_img_prob_class = test_img * prob_class_img[classidx]

    # Draw the test probability for this class
    plt.subplot(1, num_classes, classidx+1)
    plt.imshow(test_img_prob_class.reshape(28,28), cmap="gray_r", norm=norm)
    plt.title("Class "+str(classidx))

    # Average the probabilities of all pixels
    prob_class.append( np.mean(test_img_prob_class) )
    print("Probability of class {}: {:.3f}".format(classidx, prob_class[classidx]))

  plt.show()

  # Pick the largest
  Y_pred[testidx] = prob_class.index(max(prob_class))
  print("Predicted: {}  Correct: {}".format(Y_pred[testidx], test_labels[testidx]))
