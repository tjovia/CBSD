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
from sklearn.metrics import roc_curve, auc
import pylab as pl
from scipy import interp
import warnings
warnings.filterwarnings("ignore")

#initializations for the ROC curve plot
mean_tpr = 0.0
mean_fpr = np.linspace(0, 1, 150)
graph = []
    
import xlwt
# Create workbook and worksheet
wbk = xlwt.Workbook()
sheet = wbk.add_sheet('Scores')


names = ["Nearest Neighbors", "Linear SVM", "Decision Tree","Random Forest","Naive Bayes"]
classifiers = [
    KNeighborsClassifier(10),
    SVC(kernel="linear", C=0.025,probability=True),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    GaussianNB()
    ]



images = ['3.csv',"6.csv","7.csv","9.csv","10.csv","13.csv","14.csv","16.csv","18.csv","20.csv","21.csv","23.csv","25.csv","26.csv","27.csv","28.csv","30.csv","31.csv","33.csv","35.csv"]
images_labeled = ['3_labeled.csv',"6_labeled.csv","7_labeled.csv","9_labeled.csv","10_labeled.csv","13_labeled.csv","14_labeled.csv","16_labeled.csv","18_labeled.csv","20_labeled.csv","21_labeled.csv","23_labeled.csv","25_labeled.csv","26_labeled.csv","27_labeled.csv","28_labeled.csv","30_labeled.csv","31_labeled.csv","33_labeled.csv","35_labeled.csv"]
surveyor_score = [4,4,2,2,2,2,2,4,5,4,4,4,5,3,3,4,2,2,3,3]
actual = []
predicted = []

for x in range(0,len(images)):
    TEST_labeled = np.loadtxt(images_labeled[x],skiprows=1, usecols=(1,4), delimiter=',')
    TEST = np.loadtxt(images[x],skiprows=1, usecols=(0,1,2), delimiter=',')
    DATA= np.load('Finaldata.npy')

    
    #All data sets
    X = DATA[::50,0:3]
    y = DATA[::50,3];



    #scaling the data
    X = StandardScaler().fit_transform(X)

    TEST = StandardScaler().fit_transform(TEST)

    #spliting data into training and testing set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4)

    row = 0  # row counter
    col = 0  # column counter

    # iterate over classifiers
    for name, clf in zip(names, classifiers):
        
        
        print name

        #Measure Accuracy using accuracy Test
        clf.fit(X_train,y_train)
        y_pred = clf.predict(X_test)
        
        #Calculate Cross Validation Score
        t1=time.time();
        scores = cross_validation.cross_val_score(clf, X_test, y_test, cv=5)
        t2=time.time();
        CrossValidationTime=t2-t1
        print("Cross Validation Score: %0.2f " % scores.mean())
        print "Execution Time %0.2f" % CrossValidationTime
        z=[]
        for x in range(0, len(TEST)):
            z.append(clf.predict(TEST[x:x+1,0:3]))
        count0 = 0
        count1 = 0

        for y in z:
            if y==0:
                count0 = count0 + 1
            else:
                count1 = count1 + 1
        PredictedScore = count0 / float(count0 + count1)
        if PredictedScore <= 0.2:
            print "1"
        elif PredictedScore <= 0.4:
            print "2"
        elif PredictedScore <= 0.6:
            print "3"
        elif PredictedScore <= 0.8:
            print "4"
        elif PredictedScore <= 1.0:
            print "5"
        predicted.append(count0 / float(count0 + count1))
        t1=time.time();
        probas_ = clf.fit(X_train, y_train).predict_proba(X_test)
        

        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1], pos_label=1)
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)
        t2=time.time();
        ROCTime=t2-t1
        
        
        
        mean_tpr =  mean_tpr/np.max(mean_tpr)
        mean_auc = auc(mean_fpr, mean_tpr)
        graph.append(mean_fpr)
        pl.plot(mean_fpr, mean_tpr, label='%s mean roc (area = %0.2f)' % (name, mean_auc), lw=2)
        

    pl.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Random guess')
    pl.xlim([-0.05, 1.05])
    pl.ylim([-0.05, 1.05])
    pl.xlabel('False Positive Rate')
    pl.ylabel('True Positive Rate')
    pl.title('Receiver Operating Characteristic')
    pl.legend(loc="lower right")
    pl.show()    
     
    #calculate the manual percentage of necrosis
    true0 = 0
    true1 = 0

    for y in TEST_labeled[:,1:2]:
        if y==0:
            true0 = true0 + 1
        else:
            true1 = true1 + 1
    actualScore = true0 / float(true0 + true1)
    
    actual.append(true0 / float(true0 + true1))
    if actualScore <= 0.2:
        print "1"
    elif actualScore <= 0.4:
        print "2"
    elif actualScore <= 0.6:
        print "3"
    elif actualScore <= 0.8:
        print "4"
    elif actualScore <= 1.0:
        print "5"

    
