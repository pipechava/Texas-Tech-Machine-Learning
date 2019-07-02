#Optional 3b

import pickle
import numpy as np
import matplotlib.pyplot as plt
import sklearn.preprocessing
import sklearn.linear_model

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