from subprocess import check_call

import pydotplus
from IPython.core.display import Image
from pandas import read_csv
from sklearn import metrics
from sklearn.externals.six import StringIO
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import os

from sklearn.tree import DecisionTreeClassifier, export_graphviz


def read_from_CSV(url):
    dirpart = os.path.dirname(__file__)
    names = ["age","sex","on_thyroxine","query_thyroxine"
             ,"antithyroid","sick","pregnant","thyroid_surgery"
             ,"I131_treatment","query_hypothyroid","query_hyperthyroid"
             ,"lithium","goitre","tumor","hypopituitary","psych"
             ,"TSH_measured","TSH","T3_measured","T3","TT4_measured"
             ,"TT4","T4U_measured","T4U","FTI_measured","FTI"
             ,"TBG_measured","TBG","referral","class"]
    dataset = read_csv(dirpart + "/DataSet/" + url,names=names)
    return dataset

def detail(dataSet):
    print(dataSet.describe)
    print(dataSet.groupby("class").size())
if __name__ == '__main__':
    print("Main")
    dirpath = os.path.dirname(__file__)
    dataSet = read_csv(dirpath + "/DataSet/" + "allhyper2Data.csv")
    testSet = read_csv(dirpath + "/DataSet/" + "allhyper2test.csv")
    #detail(dataSet)
    data_array = dataSet.values
    test_array = testSet.values
    #X = X.astype("float")
    X = data_array[:,0:29]
    Y = data_array[:,29]
    #X = X.astype("float64")
    #Y = Y.astype("float64")

    #print(X)
    #print(Y)

    X_test = data_array[:,0:29]
    Y_test = data_array[:,29]

    #print(X[-1])

    #print(X[:,28:29])
    labelencoder_y = LabelEncoder()
    #X[:28,29] = labelencoder_y.fit_transform(X[:,28,29])

    #X[:,28:29] = labelencoder_y.fit_transform(X[:,28,29])
    #print(Y)
    Y = labelencoder_y.fit_transform(Y)
    #print(Y)
    Y_test = labelencoder_y.fit_transform(Y_test)

    X_train,X_val,Y_train,Y_val = train_test_split(X,Y,test_size=0.2,random_state=1)

    clf = DecisionTreeClassifier()
    clf = clf.fit(X_train,Y_train)
    Y_pred = clf.predict(X_test)

    print(confusion_matrix(Y_test, Y_pred))

    print("Accuracy:", metrics.accuracy_score(Y_test, Y_pred))

    #try to creat node
    dot_data = StringIO()
    export_graphviz(clf,out_file='tree_limited.dot',filled=True,rounded=True,special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    check_call(['dot', '-Tpng', dirpath + "/tree_limited.dot", '-o', 'tree_limited.png'])
    Image(filename = dirpath + '/tree_limited.png')


    # classifier = RandomForestClassifier()
    # classifier = classifier.fit(X_train,Y_train)
    # predicted = classifier.predict(X_test)
    #
    # print("Confusion Matrix : ")
    # print(confusion_matrix(Y_test,predicted))
    # print("Accuracy Score : ",accuracy_score(Y_test,predicted))
    # print("Report : ")
    # print(classification_report(Y_test,predicted))
