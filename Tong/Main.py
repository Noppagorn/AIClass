import re
from subprocess import check_call

import pydotplus
from IPython.core.display import Image
from pandas import read_csv
from sklearn import metrics
from sklearn.externals.six import StringIO
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import os
import numpy as np

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

class ModelClassification:
    def __init__(self):
        print("model has create")
        dirpath = os.path.dirname(__file__)
        self.names = ["age","sex","on_thyroxine","query_thyroxine"
             ,"antithyroid","sick","pregnant","thyroid_surgery"
             ,"I131_treatment","query_hypothyroid","query_hyperthyroid"
             ,"lithium","goitre","tumor","hypopituitary","psych"
             ,"TSH_measured","TSH","T3_measured","T3","TT4_measured"
             ,"TT4","T4U_measured","T4U","FTI_measured","FTI"
             ,"TBG_measured","TBG","referral","class"]
        self.dataSet = read_csv(dirpath + "/DataSet/" + "allhyper2Data.csv",header=0)
        self.testSet = read_csv(dirpath + "/DataSet/" + "allhyper2Test.csv",header=0)

    def preprocess(self,url,outName):
        print("preprocessing")
        parth = os.path.dirname(__file__)
        names = ["age", "sex", "on_thyroxine", "query_thyroxine"
            , "antithyroid", "sick", "pregnant", "thyroid_surgery"
            , "I131_treatment", "query_hypothyroid", "query_hyperthyroid"
            , "lithium", "goitre", "tumor", "hypopituitary", "psych"
            , "TSH_measured", "TSH", "T3_measured", "T3", "TT4_measured"
            , "TT4", "T4U_measured", "T4U", "FTI_measured", "FTI"
            , "TBG_measured", "TBG", "referral", "class"]
        dataset = read_csv(parth + "/DataSet/" + url, names=names)

        # option 1
        dataset['class'] = [re.sub(r'\.\|.*', '', str(x)) for x in dataset['class']]
        dataset['class'] = [self.mapTransfromClass(x) for x in dataset['class']]

        dataset.loc[dataset['sex'] == "M", 'sex'] = 1
        dataset.loc[dataset['sex'] == "F", 'sex'] = 0
        dataset = dataset.replace("t", 1)
        dataset = dataset.replace('f', 0)
        dataset = dataset.replace("?", -1)
        print(dataset.groupby("referral").size())
        dataset['referral'] = [self.mapTransfromRef(x) for x in dataset['referral']]
        dataset = dataset.set_index("age")
        # le = preprocessing.LabelEncoder()
        # dataset['referral'] = le.fit_transform(dataset['referral'])

        print(dataset.head(5))
        # print(dataset.groupby("referral").size())
        # print(dataset.head(10))
        # print(dataset.groupby("class").size())

        # option 2
        # df['team'] = df['team'].apply(lambda x: re.sub(r'[\n\r]*', '', str(x)))

        dataset.to_csv("DataSet/" + outName, encoding="utf-8")
    def trainAndPredict(self):
        data_array = self.dataSet.values
        test_array = self.testSet.values

        X = data_array[:, 0:29]
        Y = data_array[:, 29]

        X_test = data_array[:, 0:29]
        Y_test = data_array[:, 29]
        Y_test = np.nan_to_num(Y_test)

        # labelencoder_y = LabelEncoder()
        # Y = labelencoder_y.fit_transform(Y)
        # Y_test = labelencoder_y.fit_transform(Y_test)

        clf = DecisionTreeClassifier(criterion="entropy", max_depth=2)
        clf = clf.fit(X, Y)
        Y_pred = clf.predict(X_test)

        print(confusion_matrix(Y_test, Y_pred))

        print("Accuracy:", metrics.accuracy_score(Y_test, Y_pred))
    def mapTransfromClass(self,x):
        wordDict = {
            "negative": 0,
            "T3 toxic": 1,
            "goitre": 2,
            "hyperthyroid": 3,
        }
        return wordDict.get(x)

    def mapTransfromRef(self,x):
        wordDict = {
            "STMW": 0,
            "SVHC": 1,
            "SVHD": 2,
            "SVI": 3,
            "other": 4
        }
        return wordDict.get(x)


if __name__ == '__main__':
    print("Main")
    dirpath = os.path.dirname(__file__)
    names = ["age","sex","on_thyroxine","query_thyroxine"
             ,"antithyroid","sick","pregnant","thyroid_surgery"
             ,"I131_treatment","query_hypothyroid","query_hyperthyroid"
             ,"lithium","goitre","tumor","hypopituitary","psych"
             ,"TSH_measured","TSH","T3_measured","T3","TT4_measured"
             ,"TT4","T4U_measured","T4U","FTI_measured","FTI"
             ,"TBG_measured","TBG","referral","class"]
    dataSet = read_csv(dirpath + "/DataSet/" + "allhyper2Data.csv",header=0)
    testSet = read_csv(dirpath + "/DataSet/" + "allhyper2Test.csv",header=0)
    #print(dataSet.head(5))
    print(dataSet.info())
    print(testSet.info())
    data_array = dataSet.values
    test_array = testSet.values

    X = data_array[:, 0:29]
    Y = data_array[:, 29]
    print(dataSet.groupby("class").size())

    X_test = test_array[:, 0:29]
    Y_test = test_array[:, 29]
    Y_test = np.nan_to_num(Y_test)

    # labelencoder_y = LabelEncoder()
    # Y = labelencoder_y.fit_transform(Y)
    # Y_test = labelencoder_y.fit_transform(Y_test)

    #clf = DecisionTreeClassifier(criterion="entropy", max_depth=2)
    clf = BernoulliNB()
    scores = cross_val_score(clf, X, Y, cv=10)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    #clf = clf.fit(X, Y)
    #Y_pred = clf.predict(X_test)
    #print(Y_pred)
    #print(np.isnan(Y_test))
    #print(np.where(np.isnan(Y_test)))

    #print(confusion_matrix(Y_test, Y_pred))
    #print("Accuracy:", metrics.accuracy_score(Y_test, Y_pred))

    # dot_data = StringIO()
    # export_graphviz(clf,out_file='tree_limited.dot',filled=True,rounded=True,special_characters=True,feature_names=names[0:29])
    # graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    # check_call(['dot', '-Tpng', dirpath + "/tree_limited.dot", '-o', 'tree_limited.png'])
    # Image(filename = dirpath + '/tree_limited.png')

