# Matthew Phelan and Trent Woods
# CS 371
# ML_updated

# References:
# https://mikulskibartosz.name/f1-score-explained-d94ee90dec5b
# https://matplotlib.org/mpl_toolkits/mplot3d/tutorial.html (help on 3D graphs) ^
import pandas as pd
import numpy as np
import csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score
import sys
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec

# to plot F1 score in our 3D figure
def F1_score_arr(prec, rec):
    F1_array = list()
    for x, y in zip(prec, rec) :
        F1_array.append((2 * (x * y) / (x + y)))
    return F1_array

df = pd.read_csv("data.csv")
# header
columns_list = ['sport', 'dport', 'proto', 'flow_id', 'count', 'len', 'ttl', 'time', 'label']
df.columns = columns_list

# features to use in teaching
features = ['sport', 'dport', 'proto', 'count', 'len', 'ttl', 'time']

# X data is used to predict, y is answer
X = df[features]
y = df['label']
# initialize lists for keeping track of data
tree_acc = list()
neural_acc = list()
SVC_acc = list()
tree_F1_ave = list()
neural_F1_ave = list()
SVC_F1_ave = list()
tree_prec_list = list()
neural_prec_list = list()
SVC_prec_list = list()
tree_rec_list = list()
neural_rec_list = list()
SVC_rec_list = list()
acc_scores = 0
# 10 iterations for accuracy
for i in range(0, 10):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    #Decision Trees
    clf_tree = tree.DecisionTreeClassifier()
    clf_tree.fit(X_train, y_train)
    # Neural network (Multilayer Perceptron Classifier)
    clf_neural = MLPClassifier()
    clf_neural.fit(X_train, y_train)
    #SVC's
    clf_SVC = SVC(gamma='scale')     #SVC USE THIS
    #clf = LinearSVC()  #Linear SVC
    clf_SVC.fit(X_train, y_train)
    tree_predict = clf_tree.predict(X_test)
    neural_predict = clf_neural.predict(X_test)
    SVC_predict = clf_SVC.predict(X_test)
    #here you are supposed to calculate the evaluation measures indicated in the project proposal (accuracy, F-score etc)
    result_tree = clf_tree.score(X_test, y_test)  #accuracy score
    result_neural = clf_neural.score(X_test, y_test)  #accuracy score
    result_SVC = clf_SVC.score(X_test, y_test)  #accuracy score
    # Save accuracy scores
    tree_acc.append(result_tree)
    neural_acc.append(result_neural)
    SVC_acc.append(result_SVC)
    # F1 Scores
    F1_tree = f1_score(y_test, tree_predict, average='weighted', labels=np.unique(tree_predict))
    F1_neural = f1_score(list(y_test), neural_predict, average='weighted', labels=np.unique(neural_predict))
    F1_SVC = f1_score(list(y_test), SVC_predict, average='weighted', labels=np.unique(SVC_predict))
    # Save F1 scores
    tree_F1_ave.append(F1_tree)
    neural_F1_ave.append(F1_neural)
    SVC_F1_ave.append(F1_SVC)
    # Precision & Recall Scores
    tree_prec = precision_score(y_test, tree_predict, average='weighted', labels=np.unique(tree_predict))
    neural_prec = precision_score(y_test, neural_predict, average='weighted', labels=np.unique(neural_predict))
    SVC_prec = precision_score(y_test, SVC_predict, average='weighted', labels=np.unique(SVC_predict))
    tree_prec_list.append(tree_prec)
    neural_prec_list.append(neural_prec)
    SVC_prec_list.append(SVC_prec)
    # Recall
    tree_recall = recall_score(y_test, tree_predict, average='weighted', labels=np.unique(tree_predict))
    neural_recall = recall_score(y_test, neural_predict, average='weighted', labels=np.unique(neural_predict))
    SVC_recall = recall_score(y_test, SVC_predict, average='weighted', labels=np.unique(SVC_predict))
    tree_rec_list.append(tree_recall)
    neural_rec_list.append(neural_recall)
    SVC_rec_list.append(SVC_recall)

''' print('Decision tree result: ', i, ' ',  result_tree)
    print('Neural Network result: ', i, ' ', result_neural)
    print('SVC Result: ', i, ' ',  result_SVC)'''

#-------------------Stacked Bar Plot of accuracies--------------------------
plt.figure(1)
ind = np.arange(len(tree_acc))   
width = 0.25

p1 = plt.bar(ind, tree_acc, width)
p2 = plt.bar(ind, neural_acc, width)
p3 = plt.bar(ind, SVC_acc, width)

plt.ylabel('Accuracy')
plt.xlabel('Executions')
plt.title('Accuracy Scores for Different ML Algorithms')
plt.xticks(ind, ('1', '2', '3', '4', '5', '6', '7', '8', '9', '10'))
plt.yticks(np.arange(0, 1.2, step=0.2))
plt.legend((p1[0], p2[0], p3[0]), ('Decision Tree', 'Neural Network', 'SVC'))
plt.tight_layout()
#-----------------------------------------------------------------------------

#---------------------------subplots for port frequencies in relation to label #---------------
plt.subplot(221)
plt.title("Label = 1")
plt.xlabel('Source Port #')
plt.ylabel('Frequency')
array1 = df.query('label == 1')
array1['sport'].value_counts()[:3].plot(kind =
                                        'bar')

plt.subplot(222)
plt.title("Label = 2")
plt.xlabel('Source Port #')
plt.ylabel('Frequency')
array2 = df.query('label == 2')
array2['sport'].value_counts()[:3].plot(kind =
                                        'bar')

plt.subplot(223)
plt.title("Label = 3")
plt.xlabel('Source Port #')
plt.ylabel('Frequency')
array3 = df.query('label == 3')
array3['sport'].value_counts()[:3].plot(kind =
                                        'bar')

plt.subplot(224)
plt.title("Label = 4")
plt.xlabel('Source Port #')
plt.ylabel('Frequency')
array4 = df.query('label == 4')
array4['sport'].value_counts()[:3].plot(kind =
                                        'bar')
plt.tight_layout()

# destination port plot
plt.figure(3)
plt.subplot(221)
plt.title("Label = 1")
plt.xlabel('Destination Port #')
plt.ylabel('Frequency')


array1 = df.query('label == 1')
array1['dport'].value_counts()[:3].plot(kind =
                                        'bar')
plt.subplot(222)
plt.title("Label = 2")
plt.xlabel('Destination Port #')
plt.ylabel('Frequency')
array2 = df.query('label == 2')
array2['dport'].value_counts()[:3].plot(kind =
                                        'bar')

plt.subplot(223)
plt.title("Label = 3")
plt.xlabel('Destination Port #')
plt.ylabel('Frequency')
array3 = df.query('label == 3')
array3['dport'].value_counts()[:3].plot(kind =
                                        'bar')
plt.subplot(224)
plt.title("Label = 4")
plt.xlabel('Destination Port #')
plt.ylabel('Frequency')
array4 = df.query('label == 4')
array4['dport'].value_counts()[:3].plot(kind =
                                        'bar')
plt.tight_layout()
#-----------------------------------------------------------------------------------

#--------------------Protocol Number and Label # tuples--------------------------
plt.figure(4)
plt.xlabel('Label #, Protocol #')
plt.ylabel('Frequency')
df.groupby('label')['proto'].value_counts().plot(kind = 'bar', title = 'Protocol # in Relation to Label #')
plt.tight_layout()
#------------------------------------------------------------------------------

#--------------------Ave TTL for each Label------------------------------------
plt.figure(5)
plt.xlabel('Label #')
plt.ylabel('Average TTL')
df.groupby('label')['ttl'].agg(np.mean).plot(kind = 'bar', title = 'Average TTL for Each Label')
plt.tight_layout()
#------------------------------------------------------------------------------

#------------------mean count rate per label-------------------------------------
plt.figure(6)
df.groupby('label')['count'].agg(np.mean).plot(kind = 'bar', title = 'Average Count for each Label')
plt.xlabel('Label #')
plt.ylabel('Average Count')
plt.tight_layout()
#--------------------------------------------------------------------------------

#--------------------mean length per label----------------------------------------
plt.figure(7)
df.groupby('label')['len'].agg(np.mean).plot(kind = 'bar', title = 'Average Length for each Label')
plt.xlabel('Label #')
plt.ylabel('Average Length')
plt.tight_layout()
#----------------------------------------------------------------------------------

#---------------------mean time per label--------------------------------------------
plt.figure(8)
df.groupby('label')['time'].agg(np.mean).plot(kind = 'bar', title = 'Average Time for each Label')
plt.xlabel('Label #')
plt.ylabel('Average Time')
plt.tight_layout()
#----------------------------------------------------------------------------------------

#-------------------F1 Scores Stacked Barplot--------------------------
plt.figure(9)
ind = np.arange(len(tree_F1_ave))
width = 0.25

p1 = plt.bar(ind, tree_F1_ave, width)
p2 = plt.bar(ind, neural_F1_ave, width)
p3 = plt.bar(ind, SVC_F1_ave, width)

plt.ylabel('Ave F1 Score')
plt.xlabel('Execution #')
plt.title('Ave F1 Scores for Each Execution ')
plt.xticks(ind, ('1', '2', '3', '4', '5', '6', '7', '8', '9', '10'))
plt.yticks(np.arange(0, 1.2, step=0.2))
plt.legend((p1[0], p2[0], p3[0]), ('Decision Tree', 'Neural Network', 'SVC'))
plt.tight_layout()
#-----------------------------------------------------------------------------


# Help for this from:
# https://mikulskibartosz.name/f1-score-explained-d94ee90dec5b
# https://matplotlib.org/mpl_toolkits/mplot3d/tutorial.html
#--------------------------Precision/Recall Graphs-----------------------------
F0 = plt.figure(10)
F0.suptitle('Decision Tree Recall vs Precision vs F1Score', fontsize=16)
plt3 = plt.subplot(projection='3d')
plt3.plot_trisurf(tree_prec_list, tree_rec_list, F1_score_arr(tree_prec_list, tree_rec_list), cmap=plt.cm.magma, linewidth=0.3)
plt3.view_init(10, -150)
plt3.set_xlabel('Precision')
plt3.set_ylabel('Recall')
plt3.set_zlabel('F1Score')

F1 = plt.figure(11)
F1.suptitle('Neural Network Recall vs Precision vs F1Score', fontsize=16)
plt3 = plt.subplot(projection='3d')
plt3.plot_trisurf(neural_prec_list, neural_rec_list, F1_score_arr(neural_prec_list, neural_rec_list), cmap=plt.cm.magma, linewidth=0.3)
plt3.view_init(10, -150)
plt3.set_xlabel('Precision')
plt3.set_ylabel('Recall')
plt3.set_zlabel('F1Score')

F2 = plt.figure(12)
F2.suptitle('SVC Recall vs Precision vs F1Score', fontsize=16)
plt3 = plt.subplot(projection='3d')
plt3.plot_trisurf(SVC_prec_list, SVC_rec_list, F1_score_arr(SVC_prec_list, SVC_rec_list), cmap=plt.cm.magma, linewidth=0.3)
plt3.view_init(10, -150)
plt3.set_xlabel('Precision')
plt3.set_ylabel('Recall')
plt3.set_zlabel('F1Score')
#----------------------------------------------------------------------------

plt.show()
sys.exit()
