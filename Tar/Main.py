import os
import re

from pandas import read_csv
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn import  metrics
if __name__ == '__main__':
    print("Start")
    dirpath = os.path.dirname(__file__)
    filename = "Autism-Adult-Data.arff"
    names = ["A1","A2","A3","A4","A5","A6","A7","A8","A9","A10"
             ,"age","gender","ethnicity","jundice","austim","contry_of_res"
             ,"used_app_before","result","age_desc","relation","class"]

    dataset = read_csv(dirpath + "/DataSet/" + filename,names=names)
    #next(dataset)
    #next(dataset) # this skip the first line.
    #print(dataset.head(20))
    #dataset = dataset.replace("?",0)
    #print(dataset.head(20))
    le = preprocessing.LabelEncoder()
    dataset["gender"] = le.fit_transform(dataset["gender"])
    print(dataset.groupby("ethnicity").size())
    dataset['ethnicity'] = [re.sub(r'\?','Others', str(x)) for x in dataset['ethnicity']]
    dataset['ethnicity'] = [re.sub(r'Others','others', str(x)) for x in dataset['ethnicity']]
    dataset['ethnicity'] = [re.sub(r'<','', str(x)) for x in dataset['ethnicity']]
    dataset["ethnicity"] = le.fit_transform(dataset["ethnicity"])

    dataset['jundice'] = [re.sub(r'\?','no', str(x)) for x in dataset['jundice']]
    dataset["jundice"] = le.fit_transform(dataset["jundice"])

    dataset['austim'] = [re.sub(r'\?','no', str(x)) for x in dataset['austim']]
    dataset["austim"] = le.fit_transform(dataset["austim"])

    dataset["contry_of_res"] = [re.sub(r'\?','others', str(x)) for x in dataset['contry_of_res']]
    dataset["contry_of_res"] = le.fit_transform(dataset["contry_of_res"])

    dataset['used_app_before'] = [re.sub(r'\?','no', str(x)) for x in dataset['used_app_before']]
    dataset["used_app_before"] = le.fit_transform(dataset["used_app_before"])

    dataset['age_desc'] = [re.sub(r'\?','\'18 and more\'', str(x)) for x in dataset['age_desc']]
    dataset["age_desc"] = le.fit_transform(dataset["age_desc"])

    dataset['relation'] = [re.sub(r'\?','Self', str(x)) for x in dataset['relation']]
    dataset["relation"] = le.fit_transform(dataset["relation"])

    dataset = dataset.replace("?",0)
    dataset.to_csv("newData.csv",encoding="utf-8")
    #print(dataset.head(20))
    #next(dataset)
    #print(dataset.head(20))
    #print(dataset.values)

    arr = dataset.values
    X = arr[:,0:20]
    Y = arr[:,20]
    Y = le.fit_transform(Y)
    print(X)

    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=1)

    model = DecisionTreeClassifier(criterion="entropy",max_depth=2)
    model = model.fit(X_train,Y_train)
    Y_pred = model.predict(X_test)
    print(confusion_matrix(Y_test,Y_pred))

    print("Accuracy score :",metrics.accuracy_score(Y_test,Y_pred))
