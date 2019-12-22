from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import os

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
    dataSet = read_from_CSV("allhyper2Data.csv")
    testSet = read_from_CSV("allhyper2test.csv")
    #detail(dataSet)
    data_array = dataSet.values
    test_array = testSet.values

    X = data_array[:,0:29]
    Y = data_array[:,29]

    X_test = data_array[:,0:29]
    Y_test = data_array[:,29]

    #print(X[-1])

    print(X[:,28:29])
    labelencoder_y = LabelEncoder()
    #X[:,28:29] = labelencoder_y.fit_transform(X[:,28,29])
    #print(Y)
    #Y = labelencoder_y.fit_transform(Y)
    #print(Y)
    #Y_test = labelencoder_y.fit_transform(Y_test)

    X_train,X_val,Y_train,Y_val = train_test_split(X,Y,test_size=0.2,random_state=1)

    classifier = RandomForestClassifier()
    classifier = classifier.fit(X_train,Y_train)
    predicted = classifier.predict(X_test)

    print("Confusion Matrix : ")
    print(confusion_matrix(Y_test,predicted))
    print("Accuracy Score : ",accuracy_score(Y_test,predicted))
    print("Report : ")
    print(classification_report(Y_test,predicted))
