import os
import re
from pandas import read_csv

if __name__ == '__main__':
    url = "allhyperTest.data"
    parth = os.path.dirname(__file__)
    names = ["age","sex","on_thyroxine","query_thyroxine"
             ,"antithyroid","sick","pregnant","thyroid_surgery"
             ,"I131_treatment","query_hypothyroid","query_hyperthyroid"
             ,"lithium","goitre","tumor","hypopituitary","psych"
             ,"TSH_measured","TSH","T3_measured","T3","TT4_measured"
             ,"TT4","T4U_measured","T4U","FTI_measured","FTI"
             ,"TBG_measured","TBG","referral","class"]
    dataset = read_csv(parth + "/DataSet/" + url,names=names)

    #print(dataset.groupby("class").size())
    #for row in dataset.values:
    #    row[29] = re.sub(r"\.\|.*","",row[29])
    #    print(row[29])
    dataset['class'] = [re.sub(r'\.\|.*','', str(x)) for x in dataset['class']]
    print(dataset.values)
    dataset.to_csv("DataSet/allhyper2Test.csv",encoding="utf-8")
    #print(dataset.groupby("class").size())
    #print(dataset.groupby("age").size())
