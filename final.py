import numpy as np
import sys
from sklearn.ensemble import RandomForestClassifier

argv = sys.argv

TRAIN_FILE = argv[1]
PRELIM_CLASS_FILE = argv[2]
PREFINAL_CLASS_FILE = argv[3]
FINAL_NO_CLASS = argv[4]

train1 = np.genfromtxt(TRAIN_FILE, dtype='float64')
train2 = np.genfromtxt(PRELIM_CLASS_FILE, dtype='float64')
train3 = np.genfromtxt(PREFINAL_CLASS_FILE, dtype='float64')

data = np.concatenate((train1, train2, train3))
data_X = data[:, :-1]
data_Y = data[:, -1]

test = np.genfromtxt(FINAL_NO_CLASS)

clf = RandomForestClassifier(n_estimators=250, min_samples_split=53, min_samples_leaf=10, max_features='auto',
                             max_depth=30, criterion='entropy', n_jobs=-1, random_state=1)
clf.fit(data_X, data_Y)
train_accuracy = clf.score(data_X, data_Y)
print("Train accuracy: " + str(train_accuracy))
print("Train File Accuracy: " + str(clf.score(train1[:, :-1], train1[:, -1])))
print("Prelim File Accuracy: " + str(clf.score(train2[:, :-1], train2[:, -1])))
print("Prefinal File Accuracy: " + str(clf.score(train3[:, :-1], train3[:, -1])))


out = clf.predict(test)
out_file = open(argv[5], 'w')
for i in out:
    out_file.write(str(int(i)) + '\n')
out_file.close()
