from numpy import *
import matplotlib.pyplot as plt
from pylab import *
from sklearn import svm, datasets
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix

import pylab as pl

surveyor_score1 = [4,4,2,2,2,2,2,4,5,4,4,4,5,3,3,3,2,2,3,3,1]
surveyor_score2 = [4,4,2,2,2,3,2,4,5,5,5,3,5,4,4,4,2,2,3,3,1]



# Compute confusion matrix
cm = confusion_matrix(surveyor_score1, surveyor_score2)

print(cm)

# Show confusion matrix in a separate window
pl.matshow(cm)
pl.title('Confusion matrix')
pl.colorbar()
pl.ylabel('surveyor1 score')
pl.xlabel('surveyor2 score')
pl.show()
