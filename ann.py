#Forest Fire Predictor

'''
 Installing Theano
 pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

 Installing Tensorflow
 pip install tensorflow

 Installing Keras
 pip install --upgrade keras
'''



'''
    Data Cleaning and Preprocessing
'''

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Importing the dataset
dataset = pd.read_csv('forestfires.csv')

# plot correlation matrix
corr = dataset.corr()
fig = plt.figure(figsize = (15,15))
sns.heatmap(corr, vmax = .75, square = True)
plt.show()

#Getting Independent and Dependent(regression and categorical) Features
X = dataset.iloc[:, 0:12].values # independent: upperbound is excluded 
y = dataset.iloc[:, 12].values # dependent variable

# Encoding categorical data for independent variables 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 2] = labelencoder_X_1.fit_transform(X[:, 2]) #For month
labelencoder_X_2 = LabelEncoder()
X[:, 3] = labelencoder_X_2.fit_transform(X[:, 3]) #For weekday

onehotencoder = OneHotEncoder(categorical_features = [2])#dummy variable for month
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:] #avoid dummy variable trap 
onehotencoder = OneHotEncoder(categorical_features = [13])#dummy variable for week
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:] #avoid dummy variable trap 

y_class = y
y_reg = y
X_class = X
X_reg = X

'''
Classification
#Convert to Acres then Classify Size
Class 1.A - one acre or less;
Class 2.B - more than one acre, but less than 10 acres;
Class 3.C - 10 acres or more, but less than 100 acres;
Class 4.D - 100 acres or more, but less than 300 acres;
Class 5.E - 300 acres or more, but less than 1,000 acres;
Class 6.F - 1,000 acres or more, but less than 5,000 acres;
'''
y_class = dataset.iloc[:, 12].values
for i in range(0, len(y)):
    y_class[i] = (y_class[i]*2.47)
    if y_class[i] < 1.0:
        y_class[i] = 1
    elif y_class[i] < 10.0:
        y_class[i] = 2
    elif y_class[i] < 100.0:
        y_class[i] = 3
    elif y_class[i] < 300.0:
        y_class[i] = 4
    elif y_class[i] < 1000.0:
        y_class[i] = 5
    elif y_class[i] < 5000.0:
        y_class[i] = 6
    else:
        y_class[i] = 7

'''Encoding For Classification'''
from keras.utils import np_utils
y = np_utils.to_categorical(y)


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_class, y_class, test_size = 0.2) #classification
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reg, y_reg, test_size = 0.2) #regression

# Feature Scaling to optimize 
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_c = sc.fit_transform(X_train_c)
X_test_c = sc.transform(X_test_c)
X_train_r = sc.fit_transform(X_train_r)
X_test_r = sc.transform(X_test_r)


'''
    Creating the ANN
'''

# Importing the Keras libraries and packages to use Tensor Flow Backend
import keras
from keras.models import Sequential #For Initializing ANN
from keras.layers import Dense #For Layers of ANN

# Initialising the ANN with sequence of layers (Could use a Graph)
classifier = Sequential()
regressor = Sequential()

# Adding the input layer and the first hidden layer
# optimal nodes in hidden layer is art (Tip: choose as avg of input+output)
classifier.add(Dense(units = 17, kernel_initializer = 'uniform', activation = 'relu', input_dim = 27))
regressor.add(Dense(units = 14, kernel_initializer = 'uniform', activation = 'relu', input_dim = 27))

# Adding the hidden layers
classifier.add(Dense(units = 17, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dense(units = 17, kernel_initializer = 'uniform', activation = 'relu'))
regressor.add(Dense(units = 14, kernel_initializer = 'uniform', activation = 'relu'))
regressor.add(Dense(units = 14, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the output layer
# Probability for the outcome 
classifier.add(Dense(units = 7, kernel_initializer = 'uniform', activation = 'softmax'))
regressor.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'linear'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
regressor.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae'])

# Fitting the ANN to the Training set
classifier.fit(X_train_c, y_train_c, batch_size = 5, epochs = 500)
regressor.fit(X_train_r, y_train_r, batch_size = 5, epochs = 500)



'''
    Making predictions and evaluating the model
'''
# Predicting the Test set results
y_pred_c = classifier.predict(X_test_c)
y_pred_r = classifier.predict(X_test_r)


'''Regression'''
plt.plot(y_test_r, color = 'red', label = 'Real data')
plt.plot(y_pred_r, color = 'blue', label = 'Predicted data')
plt.title('Regression Predictions')
plt.legend()
plt.xlabel('X')
plt.ylabel('Burn Area'); 
plt.show()


'''Classification'''
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt   

y_pred_c = (y_pred_c > 0.5)
cm = confusion_matrix( y_test_c.argmax(axis=1), y_pred_c.argmax(axis=1))

ax= plt.subplot()
sns.heatmap(cm, annot=True, ax = ax);
ax.set_xlabel('Predicted Labels');ax.set_ylabel('True Labels'); 
ax.set_title('Classification Confusion Matrix'); 
ax.xaxis.set_ticklabels(['A','B','C','D','E','F']); ax.yaxis.set_ticklabels(['A','B','C','D','E','F']);



'''
===== ===== =====
Evaluating, Improving, Parameter Tuning 
===== ===== =====
'''
from keras.wrappers.scikit_learn import KerasClassifier
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from keras.models import Sequential #For Initializing ANN
from keras.layers import Dense #For Layers of ANN
def build_classifier():
    classifier = Sequential() 
    classifier.add(Dense(units = 14, kernel_initializer = 'uniform', activation = 'relu', input_dim = 27))
    classifier.add(Dense(units = 14, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 14, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 7, kernel_initializer = 'uniform', activation = 'softmax'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
def build_regressor():
    regressor = Sequential() 
    regressor.add(Dense(units = 14, kernel_initializer = 'uniform', activation = 'relu', input_dim = 27))
    regressor.add(Dense(units = 14, kernel_initializer = 'uniform', activation = 'relu'))
    regressor.add(Dense(units = 14, kernel_initializer = 'uniform', activation = 'relu'))
    regressor.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'linear'))
    regressor.compile(optimizer = 'adam', loss = 'mse', metrics=['mse', 'mae'])
    return regressor
classifier = KerasClassifier(build_fn = build_classifier, batch_size = 5, epochs = 100)
regressor = KerasRegressor(build_fn = build_regressor, batch_size = 5, epochs = 100)

accuracies_class = cross_val_score(estimator = classifier, X = X_train, y=y_train, cv = 10, n_jobs = -1)
accuracies_reg = cross_val_score(estimator = regressor, X = X_train, y=y_train, scoring='r2',cv = 10, n_jobs = 1)

mean_class = accuracies_class.mean()
variance_class = accuracies_class.std()

mean_reg = accuracies_reg.mean()
variance_reg = accuracies_reg.std()

#Dropout Regularization (Randomly Drops Out Neurons)
#Address Over Fitting After Each Layer
#from keras.layers import Dropout #For Layers of ANN
#classifier.add(Dropout(p=0.1))

#Tuning For Epochs, Batch Size, Optimizer
from keras.wrappers.scikit_learn import KerasClassifier
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential 
from keras.layers import Dense 


def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units = 14, kernel_initializer = 'uniform', activation = 'relu', input_dim = 27))
    classifier.add(Dense(units = 14, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 7, kernel_initializer = 'uniform', activation = 'softmax'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size': [5, 25],
              'epochs': [100, 500],
              'optimizer': ['adam', 'rmsprop']}
c_grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)
c_grid_search = c_grid_search.fit(X_train, y_train)
c_best_parameters = c_grid_search.best_params_
c_best_accuracy = c_grid_search.best_score_


def build_regressor(optimizer):
    regressor = Sequential()
    regressor.add(Dense(units = 14, kernel_initializer = 'uniform', activation = 'relu', input_dim = 27))
    regressor.add(Dense(units = 14, kernel_initializer = 'uniform', activation = 'relu'))
    regressor.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'linear'))
    regressor.compile(optimizer = optimizer, loss = 'mse', metrics = ['mse', 'mae'])
    return regressor
regressor = KerasRegressor(build_fn = build_regressor)
parameters = {'batch_size': [5, 25],
              'epochs': [100, 500],
              'optimizer': ['adam', 'rmsprop']}
r_grid_search = GridSearchCV(estimator = regressor,
                           param_grid = parameters,
                           scoring = 'r2',
                           cv = 10)
r_grid_search = r_grid_search.fit(X_train, y_train)
r_best_parameters = r_grid_search.best_params_
r_best_accuracy = r_grid_search.best_score_
