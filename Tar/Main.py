import os
from pandas import read_csv
if __name__ == '__main__':
    print("Start")
    dirpath = os.path.dirname(__file__)
    filename = "Autism-Adult-Data.arff"
    dataset = read_csv(dirpath + "/DataSet/" + filename)
    #next(dataset) # this skip the first line.