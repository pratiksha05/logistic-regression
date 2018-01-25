# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 11:12:24 2018

@author: A664120
"""

import pandas
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
import pickle
from flask import Flask

app = Flask(__name__)
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pandas.read_csv(url, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
test_size = 0.33
seed = 7
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=test_size, random_state=seed)
# Fit the model on 33%
model = LogisticRegression()
model.fit(X_train, Y_train)
# save the model to disk
filename = 'finalized_model.sav'
pickle.dump(model, open(filename, 'wb'))

# some time later...

# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
@app.route('/score')
def accuracy():
        loaded_model = pickle.load(open(filename, 'rb'))
        result = loaded_model.score(X_test, Y_test)
        print(result)
        return("accuracy:{}".format(result))
if __name__ == "__main__":
        app.run(host='0.0.0.0', port=9000)