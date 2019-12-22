import os
import re
import numpy as np
from pandas import read_csv
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

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
    dataset = dataset.replace("?",np.nan)

    print(dataset.groupby("referral").size())

    #fit = dataset.apply(lambda x: dataset.fit_transform(x))
    #OneHotEncoder().fit_transform(dataset)
    labelencoder_y = LabelEncoder()
    # X[] = labelencoder_y.fit(X[-1])

    print(dataset[:,25:26])
    # for i,row in enumerate(dataset.values):
    #     sex = row[1]
    #     if (sex == "M"):
    #         row[1] = 0
    #     else:
    #         row[1] = 1
    #option 2
    #df['team'] = df['team'].apply(lambda x: re.sub(r'[\n\r]*', '', str(x)))
    dataset.to_csv("DataSet/allhyper2test.csv",encoding="utf-8")
