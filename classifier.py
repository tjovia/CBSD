'''
Jovia Tuhaise
'''
print __doc__
import numpy as np
import pylab as pl
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.lda import LDA
from sklearn.qda import QDA
from sklearn import cross_validation
from sklearn import preprocessing
import time
from sklearn.metrics import accuracy_score


import warnings
warnings.filterwarnings("ignore")
    
import xlwt
# Create workbook and worksheet
wbk = xlwt.Workbook()
sheet = wbk.add_sheet('Scores')


names = [ "Naive Bayes","Nearest Neighbors","Linear SVM","RBF SVM", "Decision Tree"]
#names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Decision Tree","Extremely Randomized Trees", "Naive Bayes"]
classifiers = [
    KNeighborsClassifier(10),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    DecisionTreeClassifier(max_depth=5),
    #ExtraTreesClassifier(n_estimators=10, max_depth=None, min_samples_split=1, random_state=0),
    #RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    GaussianNB()
    ]
'''
#Loading Data and spliting data into Training, Test sets
DATA = np.loadtxt('allTestData.csv',skiprows=1, 
               usecols=(0,1,2,4), delimiter=',')
'''

DATA= np.load('Finaldata.npy')

print DATA.shape
#All data sets
X = DATA[::50,0:3]
y = DATA[::50,3];

print X.shape
print y.shape




X = StandardScaler().fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4)

row = 0  # row counter
col = 0  # column counter

# iterate over classifiers
for name, clf in zip(names, classifiers):
    
    #print name
    print name

    #Measure Accuracy using accuracy Test
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    accuracyScore = accuracy_score(y_test, y_pred)
    print "Accuracy Score %0.2f" % accuracyScore
    #print ""

    #Calculate Cross Validation Score
    t1=time.time();
    scores = cross_validation.cross_val_score(clf, X_test, y_test, cv=5)
    t2=time.time();
    t3=t2-t1
    
     
    print("Accuracy: %0.2f " % scores.mean())
    print "Execution Time %0.2f" % t3
    print ""
    '''
    #write the scores to excel
    #sheet.write(row,col,name)
    row += 1
    for x in scores:
        print x

    fit = clf.fit(X_train, y_train)
    col += 1
    row=0
    print "\n"
'''
