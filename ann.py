#Forest Fire Predictor

'''
 Installing Theano
 pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

 Installing Tensorflow
 pip install tensorflow

 Installing Keras
 pip install --upgrade keras
'''

# Importing the libraries
# Importing the Keras libraries and packages to use Tensor Flow Backend
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import keras
from keras.models import Sequential #For Initializing ANN
from keras.layers import Dense #For Layers of ANN
from keras.layers import Dropout #For Layers of ANN
from keras.wrappers.scikit_learn import KerasClassifier
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')

'''
=== === ===
Classification
=== === ===
'''


'''
    Data Cleaning and Preprocessing
'''

# Importing the dataset
dataset_class = pd.read_csv('forestfires.csv')

#Getting Independent and Dependent(regression and categorical) Features
X_Class = dataset_class.iloc[:, 0:12].values # independent: upperbound is excluded 
y_Class = dataset_class.iloc[:, 12].values # dependent variable

# Encoding categorical data for independent variables 
labelencoder_X_1 = LabelEncoder()
X_Class[:, 2] = labelencoder_X_1.fit_transform(X_Class[:, 2]) #For month
labelencoder_X_2 = LabelEncoder()
X_Class[:, 3] = labelencoder_X_2.fit_transform(X_Class[:, 3]) #For weekday

onehotencoder = OneHotEncoder(categorical_features = [2])#dummy variable for month
X_Class = onehotencoder.fit_transform(X_Class).toarray()
X_Class = X_Class[:, 1:] #avoid dummy variable trap 
onehotencoder = OneHotEncoder(categorical_features = [13])#dummy variable for week
X_Class = onehotencoder.fit_transform(X_Class).toarray()
X_Class = X_Class[:, 1:] #avoid dummy variable trap 

'''
#Convert to Acres then Classify Size
Class 1.A - one acre or less;
Class 2.B - more than one acre, but less than 10 acres;
Class 3.C - 10 acres or more, but less than 100 acres;
Class 4.D - 100 acres or more, but less than 300 acres;
Class 5.E - 300 acres or more, but less than 1,000 acres;
Class 6.F - 1,000 acres or more, but less than 5,000 acres;
'''
for i in range(0, len(y_Class)):
    y_Class[i] = (y_Class[i]*2.47)
    if y_Class[i] < 1.0:
        y_Class[i] = 1
    elif y_Class[i] < 10.0:
        y_Class[i] = 2
    elif y_Class[i] < 100.0:
        y_Class[i] = 3
    elif y_Class[i] < 300.0:
        y_Class[i] = 4
    elif y_Class[i] < 1000.0:
        y_Class[i] = 5
    elif y_Class[i] < 5000.0:
        y_Class[i] = 6
    else:
        y_Class[i] = 7

y_Class_Corrected = y_Class.astype(int)

# Splitting the dataset into the Training set and Test set
X_train_C, X_test_C, y_train_C, y_test_C = train_test_split(X_Class, y_Class_Corrected, test_size = 0.2) #classification

# Feature Scaling to optimize 
sc = StandardScaler()
X_train_C = sc.fit_transform(X_train_C)
X_test_C = sc.transform(X_test_C)

#Plot correlation matrix
corr = dataset_class.corr()
fig = plt.figure(figsize = (15,15))
sns.heatmap(corr, vmax = .75, square = True)
plt.show()

#Dimensionality Reduction
#Feature Elimination
#Evaluate statistical significance of each feature using Backward Elimination
'''
import statsmodels.formula.api as sm
X_Class = np.append(arr = np.ones((517, 1)).astype(int), values = X_Class, axis = 1)

def backwardElimination(x, sl):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
    return x

#Significance Level = 15%
y_temp = dataset_class.iloc[:, 12].values
SL = 0.15
X_opt = X_Class[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 
              15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]]
X_Modeled = backwardElimination(X_opt, SL)
regressor_OLS = sm.OLS(endog = y_temp, exog = X_Modeled).fit()
regressor_OLS.summary()
'''

#Feature Selection
#PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 2) #Replace None by 2
#21% Variance Explained with 2
X_train_C = pca.fit_transform(X_train_C)
X_test_C = pca.transform(X_test_C)
explained_variance = pca.explained_variance_ratio_ 

'''
    Parameter Tuning 
'''

#Tuning For Epochs, Batch Size, Optimizer
def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units = 10, kernel_initializer = 'uniform', activation = 'relu', input_dim = 4))
    classifier.add(Dense(units = 10, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 10, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size': [1, 16, 32],
              'epochs': [100, 500],
              'optimizer': ['adam', 'rmsprop']}
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)
grid_search = grid_search.fit(X_train_C, y_train_C)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_

print('Best Parameters: %s' % best_parameters)
print('Best Accuracy: %s' % best_accuracy)

'''
    Creating the ANN
'''

# Initialising the ANN with sequence of layers (Could use a Graph)
classifier = Sequential()

# Adding the input layer and the first hidden layer
# optimal nodes in hidden layer is art (Tip: choose as avg of input+output)
classifier.add(Dense(units = 20, kernel_initializer = 'uniform', activation = 'relu', input_dim = 2))

# Adding the hidden layers
classifier.add(Dense(units = 10, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dropout(rate=0.5))
classifier.add(Dense(units = 10, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dropout(rate=0.5))

# Adding the output layer
# Probability for the outcome 
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train_C, y_train_C, batch_size = 5, epochs = 100)

'''
    Making predictions and evaluating the model
'''

# Predicting the Test set results
# Making the Confusion Matrix
y_pred_C = classifier.predict(X_test_C)
cm = confusion_matrix(y_test_C,y_pred_C)

ax= plt.subplot()
sns.heatmap(cm, annot=True, ax = ax);
ax.set_xlabel('Predicted Labels');ax.set_ylabel('True Labels'); 
ax.set_title('Classification Confusion Matrix'); 
ax.xaxis.set_ticklabels(['A','B','C','D','E','F']); ax.yaxis.set_ticklabels(['A','B','C','D','E','F']);

correct = 0
total = 0
for i in range(0, len(cm)):
    for j in range(0, len(cm)):
        if(i==j):
            correct = correct + cm[i][j]
        total = total + cm[i][j]
        
Accuracy = correct/total
print('Accuracy: %.2f%%' % (Accuracy*100))

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test_C, y_test_C
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('tomato', 'lightgreen', 'teal', 'slategray', 'orchid', 'pink')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('tomato', 'lightgreen', 'teal', 'slategray', 'orchid', 'pink'))(i), label = j)
plt.title('ANN (Test set)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()

#Evaluation
from sklearn.model_selection import cross_val_score

def build_classifier():
    classifier = Sequential() 
    classifier.add(Dense(units = 20, kernel_initializer = 'uniform', activation = 'relu', input_dim = 2))
    classifier.add(Dense(units = 10, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dropout(rate=0.5))
    classifier.add(Dense(units = 10, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dropout(rate=0.5))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier, batch_size = 5, epochs = 100)
accuracies_class = cross_val_score(estimator = classifier, X = X_train_C, y=y_train_C, cv = 10, n_jobs = -1)
mean_class = accuracies_class.mean()
variance_class = accuracies_class.std()

print('Mean: %s' % mean_class)
print('Variance: %s' % variance_class)


'''
=== === ===
Regression
=== === ===
'''



'''
    Data Cleaning and Preprocessing
'''

# Importing the dataset
dataset = pd.read_csv('forestfires.csv')

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

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# Feature Scaling to optimize 
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#PCA Dimensionality Reduction
pca_R = PCA(n_components = 2) #Replace None by 2
#21% Variance Explained with 2
X_train = pca_R.fit_transform(X_train)
X_test = pca_R.transform(X_test)
explained_variance_R = pca_R.explained_variance_ratio_ 

'''
Parameter Tuning 
'''
def build_regressor(optimizer):
    regressor = Sequential()
    regressor.add(Dense(units = 4, kernel_initializer = 'uniform', activation = 'relu', input_dim = 2))
    regressor.add(Dense(units = 4, kernel_initializer = 'uniform', activation = 'relu'))
    regressor.add(Dense(units = 4, kernel_initializer = 'uniform', activation = 'relu'))
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

'''
    Creating the ANN
'''

# Initialising the ANN with sequence of layers (Could use a Graph)
regressor = Sequential()

# Adding the input layer and the first hidden layer
# optimal nodes in hidden layer is art (Tip: choose as avg of input+output)
regressor.add(Dense(units = 4, kernel_initializer = 'uniform', activation = 'relu', input_dim = 2))

# Adding the hidden layers
regressor.add(Dense(units = 4, kernel_initializer = 'uniform', activation = 'relu'))
regressor.add(Dropout(rate=0.25))
regressor.add(Dense(units = 4, kernel_initializer = 'uniform', activation = 'relu'))
regressor.add(Dropout(rate=0.25))

# Adding the output layer
# Probability for the outcome 
regressor.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'linear'))

# Compiling the ANN
regressor.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae'])

# Fitting the ANN to the Training set
regressor.fit(X_train, y_train, batch_size = 5, epochs = 100)

'''
    Making predictions and evaluating the model
'''
# Predicting the Test set results
y_pred = regressor.predict(X_test)

# Visualizing Results
plt.plot(y_test, color = 'red', label = 'Real data')
plt.plot(y_pred, color = 'blue', label = 'Predicted data')
plt.title('Regression Predictions')
plt.legend()
plt.xlabel('X')
plt.ylabel('Burn Area'); 
plt.show()

#Evaluation
def build_regressor():
    regressor = Sequential() 
    regressor.add(Dense(units = 4, kernel_initializer = 'uniform', activation = 'relu', input_dim = 2))
    regressor.add(Dense(units = 4, kernel_initializer = 'uniform', activation = 'relu'))
    regressor.add(Dropout(rate=0.25))
    regressor.add(Dense(units = 4, kernel_initializer = 'uniform', activation = 'relu'))
    regressor.add(Dropout(rate=0.25))
    regressor.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'linear'))
    regressor.compile(optimizer = 'adam', loss = 'mse', metrics=['mse', 'mae'])
    return regressor

regressor = KerasRegressor(build_fn = build_regressor, batch_size = 5, epochs = 100)
accuracies_reg = cross_val_score(estimator = regressor, X = X_train, y=y_train, scoring='r2',cv = 10, n_jobs = 1)

mean_reg = accuracies_reg.mean()
variance_reg = accuracies_reg.std()

