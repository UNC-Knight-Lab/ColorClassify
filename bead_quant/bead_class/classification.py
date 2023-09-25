from tkinter.ttk import LabeledScale
import numpy as np
import pandas as pd
import glob
from sklearn.model_selection import RepeatedStratifiedKFold
from matplotlib import pyplot as plt
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler
from collections import Counter
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from scipy import stats

def classification(path):
    print("Beginning analysis...")

    train, groups, data = open_arrays(path)

    while True:
        opt = input("Do you want to optimize a gradient boosting classifier for your training data? Enter 'Y' or 'n': ")

        if str(opt) == "Y":
            opt_params = optimization(train, groups)
            n_estimators = opt_params["n_estimators"]
            learning_rate = opt_params["learning_rate"]
            subsample = opt_params["subsample"]
            max_depth = opt_params["max_depth"]
            break

        elif str(opt) == "n":
            print("Please input model parameters from a previously trained model below.")

            while True:
                estimators_input = input("Please input the number of estimators: ")
                try:
                    n_estimators = int(estimators_input)
                    break
                except ValueError:
                    print("Unexpected value. Please enter an integer value.")
            
            while True:
                learning_input = input("Please input the learning rate: ")
                try:
                    learning_rate = float(learning_input)
                    break
                except ValueError:
                    print("Unexpected value. Please enter a float value.")
            
            while True:
                subsample_input = input("Please input subsampling: ")
                try:
                    subsample = float(subsample_input)
                    break
                except ValueError:
                    print("Unexpected value. Please enter a float value.")

            while True:
                max_depth_input = input("Please input the maximum tree depth: ")
                try:
                    max_depth = int(max_depth_input)
                    break
                except ValueError:
                    print("Unexpected value. Please enter an integer value.")                
            break
        else:
            print("Unexpected input. Please try again.")
    
    while True:
        to_fit = input("Do you want to predict unknowns using this model? Enter 'Y' or 'n': ")

        if str(to_fit) == 'Y':
            while True:
                repeat_inp = input("Please input the number of iterations for model fitting: ")
                try:
                    repeat = int(repeat_inp)
                    break
                except ValueError:
                    print("Unexpected input. Please try again.")
            
            predictions = iterate(train, groups, repeat, data, learning_rate, max_depth, n_estimators, subsample)
            labels, counts = count_preds(predictions)
            plot_export(labels, counts, path)
            break

        elif str(to_fit) == "n":
            break
        else:
            print("Unexpected input. Please try again.")

    print("End of evaluation.")

    
def open_arrays(path):
    print("Importing datasets...")

    for file in glob.iglob(path + "/*_train.xlsx"):
        training_data = pd.read_excel(file, header=None)
        training_data = training_data.values

        training_groups = training_data[:,3]
        training_data = np.delete(training_data,3,1)
    
    for file in glob.iglob(path + "/*_test.xlsx"):
        test_data = pd.read_excel(file, header=None)
        test_data = test_data.values

    return training_data, training_groups, test_data

## Split training set and select model
def optimization(train, groups):

    print("Performing parameter optimization...")
    
    oversample = RandomOverSampler()
    X_over, y_over = oversample.fit_resample(train, groups)
    
    model = GradientBoostingClassifier()
    
    grid = dict()
    grid['n_estimators'] = [10,50] #[10, 50, 100, 500]
    grid['learning_rate'] = [0.1, 1.0] #[0.0001, 0.001, 0.01, 0.1, 1.0]
    grid['subsample'] = [0.5,0.7] #[0.5, 0.7, 1.0]
    grid['max_depth'] = [3,7] #[3, 7, 9]
  
    cv = RepeatedStratifiedKFold(n_splits=4, n_repeats=3, random_state=1)
    
    grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy')
    
    grid_result = grid_search.fit(X_over, y_over)
    
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

    return grid_result.best_params_

    # summarize all scores that were evaluated
    # means = grid_result.cv_results_['mean_test_score']
    # stds = grid_result.cv_results_['std_test_score']
    # params = grid_result.cv_results_['params']
    # for mean, stdev, param in zip(means, stds, params):
    #     print("%f (%f) with: %r" % (mean, stdev, param))
    
    

def predict(train, groups, data, learning_rate, max_depth, n_estimators, subsample):
    
    over = RandomOverSampler()
    model = GradientBoostingClassifier(learning_rate=learning_rate, max_depth=max_depth, n_estimators=n_estimators, subsample=subsample)
    steps = [('over',over),('model', model)]
    pipeline = Pipeline(steps=steps)
    
    pipeline.fit(train, groups)
    predictions = pipeline.predict(data)
    
    return predictions

def iterate(train, groups, repeats, data, learning_rate, max_depth, n_estimators, subsample):
    print("Beginning iterations...")
    
    all_predict = np.zeros((len(data[:,0]),repeats))
    mode_predict = []
                            
    for i in range(repeats):
        if i % 10 == 0:
            print("Repeat number ", i)

        all_predict[:,i] = predict(train, groups, data, learning_rate, max_depth, n_estimators, subsample)
    
    for i in range(len(data[:,0])):
        mode_predict.append(stats.mode(all_predict[i,:], keepdims=True)[0])
        
    return mode_predict

def count_preds(p):
    p = np.vstack(p)
    p = p.flatten()
    p = p.astype(int)
    c = Counter(p)
    
    labels = c.keys()
    counts = c.values()
    
    return labels, counts

def plot_export(labels, counts, path):
    plt.bar(labels, counts)
    plt.xlabel("Class")
    plt.ylabel("Counts")
    plt.savefig(path + "/unknown_classification.png")
    plt.show()

    results = pd.DataFrame()
    results["labels"] = labels
    results["counts"] = counts
    results.to_excel(path + "/unknown_classification.xlsx")