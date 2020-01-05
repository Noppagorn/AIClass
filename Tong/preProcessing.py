import os
import re
import numpy as np
from pandas import read_csv
from sklearn import preprocessing

def mapTransfromClass(x):
    wordDict = {
        "negative" : 0,
        "T3 toxic" : 1,
        "goitre" : 2,
        "hyperthyroid" : 3,
    }
    return wordDict.get(x)
def mapTransfromRef(x):
    wordDict = {
        "STMW" : 0,
        "SVHC" : 1,
        "SVHD" : 2,
        "SVI" : 3,
        "other" : 4
    }
    return wordDict.get(x)
def handleCSV(url,outName):
    parth = os.path.dirname(__file__)
    names = ["age","sex","on_thyroxine","query_thyroxine"
             ,"antithyroid","sick","pregnant","thyroid_surgery"
             ,"I131_treatment","query_hypothyroid","query_hyperthyroid"
             ,"lithium","goitre","tumor","hypopituitary","psych"
             ,"TSH_measured","TSH","T3_measured","T3","TT4_measured"
             ,"TT4","T4U_measured","T4U","FTI_measured","FTI"
             ,"TBG_measured","TBG","referral","class"]
    dataset = read_csv(parth + "/DataSet/" + url,names=names)


    #option 1
    dataset['class'] = [re.sub(r'\.\|.*','', str(x)) for x in dataset['class']]
    dataset['class'] = [ mapTransfromClass(x) for x in dataset['class']]

    dataset.loc[dataset['sex'] == "M", 'sex'] = 1
    dataset.loc[dataset['sex'] == "F", 'sex'] = 0
    dataset = dataset.replace("t",1)
    dataset = dataset.replace('f',0)
    dataset = dataset.replace("?",-1)
    print(dataset.groupby("referral").size())
    dataset['referral'] = [ mapTransfromRef(x) for x in dataset['referral']]
    dataset = dataset.set_index("age")
    #le = preprocessing.LabelEncoder()
    #dataset['referral'] = le.fit_transform(dataset['referral'])

    print(dataset.head(5))
    #print(dataset.groupby("referral").size())
    #print(dataset.head(10))
    #print(dataset.groupby("class").size())

    #option 2
    #df['team'] = df['team'].apply(lambda x: re.sub(r'[\n\r]*', '', str(x)))

    dataset.to_csv("DataSet/" + outName,encoding="utf-8")
if __name__ == '__main__':
    url = "allhyper.data"
    urlt = "allhyper.test"

    handleCSV(url,"allhyper2Data.csv")
    handleCSV(urlt,"allhyper2Test.csv")
