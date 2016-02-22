import csv
from sklearn.feature_extraction import DictVectorizer
from sklearn import preprocessing
from sklearn import tree
from sklearn.externals.six import StringIO

allCommonData = open('common.csv')
reader = csv.reader(allCommonData)
header = reader.next()
# print header

featureList = []
labelList = []

for row in reader:
    # print row
    labelList.append(row[len(row)-1])
    rowDict = {}
    for i in range(1,len(row)-1):
        rowDict[header[i]] = row[i]
        # print rowDict
    featureList.append(rowDict)

# print featureList

#vet
vet = DictVectorizer()
dummyX = vet.fit_transform(featureList).toarray()

lb = preprocessing.LabelBinarizer()
dummyY = lb.fit_transform(labelList)
# print dummyY
#classify
clf = tree.DecisionTreeClassifier(criterion='entropy')
clf = clf.fit(dummyX,dummyY)

with open('common.dot','w') as f:
    f = tree.export_graphviz(clf,out_file=f,feature_names=vet.get_feature_names())

oneRowX = dummyX[0,:]
print str(oneRowX)
newRowX = oneRowX
newRowX[0] =1
newRowX[2] =0
perDicedY = clf.predict(newRowX)
print "perDicedY:" + str(perDicedY)