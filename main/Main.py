import pandas as pd
import pydotplus as pdp
import os
from sklearn import metrics
from sklearn.externals.six import StringIO
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from subprocess import check_call
from IPython.core.display import Image

dir_path = os.path.dirname(__file__)

# load raw data
names = ["age", "sex", "on_thyroxine", "query_on_thyroxine",
         "on_antithyroid_medication", "sick", "pregnant", "thyroid_surgery",
         "I131_treatment", "query_hypothyroid", "query_hyperthyroid",
         "lithium", "goitre", "tumor", "hypopituitary", "psych",
         "TSH_measured", "TSH", "T3_measured", "T3", "TT4_measured",
         "TT4", "T4U_measured", "T4U", "FTI_measured", "FTI",
         "TBG_measured", "TBG", "referral_source", "class"]
data_set = pd.read_csv(dir_path + "/allhyper.data", names=names)
test_set = pd.read_csv(dir_path + "/allhyper.test", names=names)
data_list = data_set.values
test_list = test_set.values

# preprocess data
x = data_list[:, 0:29]
y = data_list[:, 29]
x_test = test_list[:, 0:29]
y_test = test_list[:, 29]

for i, line in enumerate(x):
    for j, col in enumerate(line):
        if col == 'f' or col == 'F' or col == 'WEST' or col == '?':
            x[i][j] = 0
        elif col == 't' or col == 'M' or col == 'STMW':
            x[i][j] = 1
        elif col == 'SVHC':
            x[i][j] = 2
        elif col == 'SVI':
            x[i][j] = 3
        elif col == 'SVI':
            x[i][j] = 4
        elif col == 'SVHD':
            x[i][j] = 5
        elif col == 'other':
            x[i][j] = 6
        else:
            x[i][j] = float(x[i][j])
for i, line in enumerate(y):
    y[i] = y[i].split('.')[0]

for i, line in enumerate(x_test):
    for j, col in enumerate(line):
        if col == 'f' or col == 'F' or col == 'WEST' or col == '?':
            x_test[i][j] = 0
        elif col == 't' or col == 'M' or col == 'STMW':
            x_test[i][j] = 1
        elif col == 'SVHC':
            x_test[i][j] = 2
        elif col == 'SVI':
            x_test[i][j] = 3
        elif col == 'SVI':
            x_test[i][j] = 4
        elif col == 'SVHD':
            x_test[i][j] = 5
        elif col == 'other':
            x_test[i][j] = 6
        else:
            x_test[i][j] = float(x_test[i][j])
for i, line in enumerate(y_test):
    y_test[i] = y_test[i].split('.')[0]

le = LabelEncoder()
y = le.fit_transform(y)
y_test = le.fit_transform(y_test)

# calculate result
classifier = DecisionTreeClassifier(criterion="entropy", max_depth=8)
classifier = classifier.fit(x, y)
y_predict = classifier.predict(x_test)

# display result
print(confusion_matrix(y_test, y_predict))
print(metrics.accuracy_score(y_test, y_predict))

dot_data = StringIO()
outfile = dir_path + "/tree.dot"
export_graphviz(classifier, out_file=outfile, filled=True, rounded=True, special_characters=True,
                feature_names=names[0:29])
graph = pdp.graph_from_dot_data(dot_data.getvalue())

check_call(['dot', '-Tpng', outfile, '-o', dir_path + '/tree_image.png'])
Image(filename=dir_path + '/tree_image.png')
