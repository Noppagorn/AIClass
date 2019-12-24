import os
from subprocess import check_call

import pydotplus
from IPython.core.display import Image
from pandas import read_csv
from sklearn import metrics
from sklearn.externals.six import StringIO
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier , export_graphviz
if __name__ == '__main__':
    print("Start")
    dirpath = os.path.dirname(__file__)
    print(dirpath)
    filename = "cmc.data"
    names = ["wifeAge","wifeEducate","husbandEdu",
             "numberChildren","religion","work","occupation",
             "livingIndex","mediaExposure","Contraception"]
    dataset = read_csv(dirpath + "/DataSet/" + filename,names=names)
    #print(dataset.describe())

    array = dataset.values
    X = array[:,0:9]
    Y = array[:,9]

    print(Y)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

    model = DecisionTreeClassifier(criterion="entropy",max_depth=6)
    model = model.fit(X_train,Y_train)
    Y_pred = model.predict(X_test)
    print(confusion_matrix(Y_test, Y_pred))


    print("Accuracy:", metrics.accuracy_score(Y_test, Y_pred))

    dot_data = StringIO()

    export_graphviz(model,out_file='tree_kk.dot',filled=True,rounded=True,special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    check_call(['dot', '-Tpng', dirpath + "/tree_kk.dot", '-o', 'tree_kk.png'])
    Image(filename = dirpath + '/tree_kk.png')

