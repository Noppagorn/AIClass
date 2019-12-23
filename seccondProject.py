import os
import re
import numpy as np
from pandas import read_csv
from sklearn import preprocessing

if __name__ == '__main__':
    url = "allhyper.test"
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
    dataset.loc[dataset['sex'] == "M", 'sex'] = 0
    dataset.loc[dataset['sex'] == "F", 'sex'] = 0
    dataset = dataset.replace("t",1)
    dataset = dataset.replace('f',0)
    dataset = dataset.replace("?",0)
    le = preprocessing.LabelEncoder()
    dataset['referral'] = le.fit_transform(dataset['referral'])

    print(dataset.groupby("referral").size())

    #option 2
    #df['team'] = df['team'].apply(lambda x: re.sub(r'[\n\r]*', '', str(x)))
    dataset.to_csv("DataSet/allhyper2Test.csv",encoding="utf-8")
