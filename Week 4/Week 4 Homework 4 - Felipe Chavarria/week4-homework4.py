#week4-homework4.py

#==============================================================================
#Practice 1
#==============================================================================

import pickle
import numpy as np
import matplotlib.pyplot as plt

# Load the training and test data from the Pickle file
with open("mnist_dataset_unscaled.pickle", "rb") as f:
      train_data, train_labels, test_data, test_labels = pickle.load(f)

# Plot a histogram of pixel values train data before standardization   
hist, bins = np.histogram(train_data, 50)
center = (bins[:-1] + bins[1:]) / 2
width = np.diff(bins)
plt.bar(center, hist, align='center', width=width)
plt.title("Hist of Train data pixel values")
plt.show()

# Plot a histogram of pixel values test data before standardization 
hist, bins = np.histogram(test_data, 50)
center = (bins[:-1] + bins[1:]) / 2
width = np.diff(bins)
plt.bar(center, hist, align='center', width=width)
plt.title("Hist of Test data pixel values")
plt.show()

# Scale the training and test data using a standard distribution
pixel_mean = np.mean(train_data)
pixel_std = np.std(train_data)
train_data = (train_data - pixel_mean) / pixel_std
test_data = (test_data - pixel_mean) / pixel_std

# Some information about the training dataset
print("Training dataset size: ", train_data.shape)
print("Class histogram: ")
print(np.histogram(train_labels, 10)[0])

# Some information about the test dataset
print("Test dataset size: ", test_data.shape)
print("Class histogram: ")
print(np.histogram(test_labels, 10)[0])

# Plot a histogram of pixel values train data after standardization 
hist, bins = np.histogram(train_data, 50)
center = (bins[:-1] + bins[1:]) / 2
width = np.diff(bins)
plt.bar(center, hist, align='center', width=width)
plt.title("Hist of Train data pixel values")
plt.show()

# Plot a histogram of pixel values test data after standardization 
hist, bins = np.histogram(test_data, 50)
center = (bins[:-1] + bins[1:]) / 2
width = np.diff(bins)
plt.bar(center, hist, align='center', width=width)
plt.title("Hist of Test data pixel values")
plt.show()

# Plot the first 5 digits as images
for idx in range(5):
  image = train_data[idx].reshape(28,28)
  plt.figure()
  plt.imshow(image, cmap="gray_r")
  plt.title("Label: "+str(train_labels[idx]))
plt.show()

#==============================================================================
#Optional 1b
#==============================================================================

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

#==============================================================================
#Practice 2
#==============================================================================

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

#==============================================================================
#Practice 3
#==============================================================================

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

num_classes = len(np.unique(train_labels))

# Train a Logistic Regression  classifier using multi_class = 'ovr'
model = sklearn.linear_model.LogisticRegression(\
    multi_class = 'ovr', solver='sag', tol=1e-2, max_iter = 100) 
model.fit(train_data, train_labels)

'''
# Train a Logistic Regression  classifier using multi_class = 'multinomial'
model = sklearn.linear_model.LogisticRegression(\
    multi_class = 'multinomial', solver = 'lbfgs', tol=1e-2, max_iter = 50) 
model.fit(train_data, train_labels)
'''

'''
        using OVR is harder to converge than multinomial (requires more 
        max_iter numbers to converge) and gives slightly worst accuracy than
        multinomial.
'''

# Predict the labels for all the test cases
Y_pred = model.predict(test_data)

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

# Compute the prediction accuracy against the training data
print("Against training set:")
Y_pred_training = model.predict(train_data)
print("  Accuracy:   {:.3f}".format(sklearn.metrics.accuracy_score(train_labels, Y_pred_training)))
print("  Precision:  {:.3f}".format(sklearn.metrics.precision_score(train_labels, Y_pred_training, average='weighted')))
print("  Recall:     {:.3f}".format(sklearn.metrics.recall_score(train_labels, Y_pred_training, average='weighted')))

# Explore the coefficients
print("Min coef:", np.min(model.coef_))
print("Max coef:", np.max(model.coef_))
print("Coef mean:", np.mean(model.coef_))
print("Coef stddev: ", np.std(model.coef_))

# Plot a histogram of coefficient values
hist, bins = np.histogram(model.coef_, 500)
center = (bins[:-1] + bins[1:]) / 2
width = np.diff(bins)
plt.bar(center, hist, align='center', width=width)
plt.title("Coefficient values")
plt.show()

# Display the coefficients as an image
for n in range(num_classes):
    coef_img = model.coef_[n].reshape(28, 28)
    plt.figure()
    plt.imshow(coef_img, cmap="viridis")
    plt.title("Coefficients for class "+str(n))
plt.show()

'''
OvR:
    Confusion Matrix:
    [[ 959    0    2    2    0    5    6    4    1    1]
     [   0 1113    3    1    0    1    5    1   11    0]
     [   9    9  919   20   11    4   10   11   36    3]
     [   4    0   18  921    2   20    4   12   20    9]
     [   1    2    5    3  914    0    9    2    6   40]
     [  10    2    0   40   10  769   17    7   29    8]
     [   9    3    7    2    6   20  907    1    3    0]
     [   2    7   22    5    8    2    1  949    5   27]
     [  10   14    6   21   15   27    8   11  850   12]
     [   7    8    1   13   32   13    1   25   12  897]]
    Accuracy:   0.920
    Precision:  0.920
    Recall:     0.920
      Class 0: Precision: 0.949 Recall: 0.979
      Class 1: Precision: 0.961 Recall: 0.981
      Class 2: Precision: 0.935 Recall: 0.891
      Class 3: Precision: 0.896 Recall: 0.912
      Class 4: Precision: 0.916 Recall: 0.931
      Class 5: Precision: 0.893 Recall: 0.862
      Class 6: Precision: 0.937 Recall: 0.947
      Class 7: Precision: 0.928 Recall: 0.923
      Class 8: Precision: 0.874 Recall: 0.873
      Class 9: Precision: 0.900 Recall: 0.889
    Against training set:
      Accuracy:   0.928
      Precision:  0.928
      Recall:     0.928
    Min coef: -2.053112575975616
    Max coef: 1.3606560278379483
    Coef mean: 0.00827929689962314
    Coef stddev:  0.17686318348282634
    
multinomial optimization:
    Confusion Matrix:
    [[ 958    0    1    4    1    9    3    3    1    0]
     [   0 1112    5    1    0    2    3    1   11    0]
     [   6    9  931   16   10    3   12   10   31    4]
     [   4    1   17  926    1   23    2   10   18    8]
     [   1    3    8    3  919    0    6    7    6   29]
     [   9    3    3   35    9  777   15    6   31    4]
     [   8    3    9    2    6   16  911    2    1    0]
     [   1    7   23    8    6    1    0  947    4   31]
     [  10   11    7   22    8   25   13   11  855   12]
     [   9    7    1    9   23    7    0   21    9  923]]
    Accuracy:   0.926
    Precision:  0.926
    Recall:     0.926
      Class 0: Precision: 0.952 Recall: 0.978
      Class 1: Precision: 0.962 Recall: 0.980
      Class 2: Precision: 0.926 Recall: 0.902
      Class 3: Precision: 0.903 Recall: 0.917
      Class 4: Precision: 0.935 Recall: 0.936
      Class 5: Precision: 0.900 Recall: 0.871
      Class 6: Precision: 0.944 Recall: 0.951
      Class 7: Precision: 0.930 Recall: 0.921
      Class 8: Precision: 0.884 Recall: 0.878
      Class 9: Precision: 0.913 Recall: 0.915
    Against training set:
      Accuracy:   0.940
      Precision:  0.939
      Recall:     0.940
    Min coef: -1.06282916564301
    Max coef: 0.8683054074349484
    Coef mean: -3.2630360998243248e-15
    Coef stddev:  0.13807327947677833
    
    conclusion: for this case multinomial optimization produced better results 
    than OvR and it is easier to converge since it takes less itereations
    and less time to converge.
'''

#==============================================================================
#Optional 3b
#==============================================================================

import pickle
import numpy as np
import matplotlib.pyplot as plt
import sklearn.preprocessing
import sklearn.linear_model
import sklearn.ensemble

sizes = [4, 8, "unscaled"]

for i in sizes:

    filename = "mnist_dataset_{}.pickle".format(i)
    
    # Load the training and test data from the Pickle file
    with open(filename, "rb") as f:
          train_data, train_labels, test_data, test_labels = pickle.load(f)
    
    # Scale the training and test data
    pixel_mean = np.mean(train_data)
    pixel_std = np.std(train_data)
    train_data = (train_data - pixel_mean) / pixel_std
    test_data = (test_data - pixel_mean) / pixel_std
    
    num_classes = len(np.unique(train_labels))
    
    # Train a Logistic Regression  classifier using multi_class = 'multinomial'
    model = sklearn.linear_model.LogisticRegression(\
        multi_class = 'multinomial', solver='sag', tol=1e-2, max_iter = 50) 
    model.fit(train_data, train_labels)
    
    # Predict the labels for all the test cases
    Y_pred = model.predict(test_data)
    
    print("Image rescaled to {}x{} for Logistic Regression".format(i,i))
    # Accuracy, precision & recall
    print("     Accuracy:   {:.3f}".format(sklearn.metrics.accuracy_score(test_labels, Y_pred)))
    print("     Precision:  {:.3f}".format(sklearn.metrics.precision_score(test_labels, Y_pred, average='weighted')))
    print("     Recall:     {:.3f}".format(sklearn.metrics.recall_score(test_labels, Y_pred, average='weighted')))
    
    # Train a Decision Tree classifier
    model = sklearn.ensemble.RandomForestClassifier(n_estimators = 100,min_samples_leaf = 5) 
    model.fit(train_data, train_labels)
    
    # Predict the labels for all the test cases
    Y_pred = model.predict(test_data)
    
    print("Image rescaled to {}x{} for Random Forest".format(i,i))
    # Accuracy, precision & recall
    print("     Accuracy:   {:.3f}".format(sklearn.metrics.accuracy_score(test_labels, Y_pred)))
    print("     Precision:  {:.3f}".format(sklearn.metrics.precision_score(test_labels, Y_pred, average='weighted')))
    print("     Recall:     {:.3f}".format(sklearn.metrics.recall_score(test_labels, Y_pred, average='weighted')))