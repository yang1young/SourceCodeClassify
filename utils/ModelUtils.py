import numpy as np


def precision_recall(label,predict,real):
    countSet = []
    predictSet = []
    rightSet = []
    for i in range(len(label)):
        countSet.append(0.0)
        predictSet.append(0.0)
        rightSet.append(0.0)

    for p, r in zip(predict, real):
        indexP = np.argmax(p)
        indexR = np.argmax(r)
        # name = label[int(indexR)]
        countSet[int(indexR)] += 1
        predictSet[int(indexP)] += 1
        if (int(indexR) == int(indexP)):
            rightSet[int(indexP)] += 1
    count = 0
    for i, j, k in zip(countSet, predictSet, rightSet):
        tag = str(label[count])
        if (i == 0):
            recall = '0'
        else:
            recall = str(k / i)
        if (j == 0):
            precision = '0'
        else:
            precision = str(k / j)
        print(tag + ',' + str(i) + ',' + str(precision) + ',' + str(recall))
        # result_file.write(tag+','+ str(i) +','+str(precision)+','+str(recall)+'\n')
        count += 1