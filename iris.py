import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

'''
Loading the dataset
'''

file = "iris_data.txt"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(file, names=names)

'''
Summarizingand visualising the dataset
'''
#shape
print(dataset.shape)
#head
print(dataset.head(20))
#descriptions
print(dataset.describe())
#class distribution
print(dataset.groupby('class').size())
#box and shisker plots
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
# plt.show()
#histogram univariate plots
#look for Gaussian distributions in input variables as algorithms can exploit this
dataset.hist()
# plt.show()
#multivariate plots
#these show interactions between the variables
#this can be used to spot structured relationships (correlations) between input variables
#scatter plot matrix
scatter_matrix(dataset)
# plt.show()

'''
Evaluating some algorithms to assess their accuracy of classiifying a flower based on unseen data

1. Separate out a validation dataset.
2. Set-up the test harness to use 10-fold cross validation.
3. Build 6 different models to predict species from flower measurements
4. Select the best model.

'''

#we need to know if the model we created is anyf good. This can be tested statistically.
#this means the model must be evalated using unseen data.
#we split the data into two 80% used to train and 20% as a validation dataset

#split-out validation dataset
#model_selection.train_test_split splits data
array = dataset.values
X = array[:,0:4]
Y = array[:,4]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

#test harness
#we will use a 10-fold cross validation to estimate accuracy.
#this will split our dataset into 10 parts, tain on 9 and test on 1 and repeat for all combinations of train-test splits
#the specific random seed does not matter.
#test options and evaluation metric
#using a metric of accuracy to evaluate models. This is a ratio of the number of correctly predicted instances divided by the total number of instances in the dataset multiplied by 100
seed = 7
scoring = 'accuracy'


'''
3. Evaluating 6 different algorithms
    a. Logistic regression (LR)
    b. linear disriminant analysis (LDA)
    c. K-nearest neightbours (KNN)
    d. classification and regression trees (CART)
    e. gaussian naive bayes (NB)
    f. support vector machines (SVM)

#must reset the random number seed before each run to ensure that the evaluation of each algorithm is performed using exactly the same data splits . It ensures the results are directly comparable.

'''
#spot check algorithms
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))

#evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

#comparing algorithms
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

#making predictions
#the SVM model had the highest accuracy
#an independent final check of the best model is required
#this can be conducted using the validation set
#we can run the SVN model dierctly on the validation set and summarize the results as a final accuracy score, a confusion matrix and a classification report

svm = SVC()
svm.fit(X_train, Y_train)
predictions = svm.predict(X_validation)
print('\n')
print(accuracy_score(Y_validation, predictions))
#the confusion matrix provides an indication of the three errors made. Finally, the classification report provides a breakdown of each class by precision, recall, f1-score and support.
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))













