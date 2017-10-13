import numpy as np
from prettytable import PrettyTable

def precision_recall(label,predict,real):
    countSet = []
    predictSet = []
    rightSet = []

    for i in range(len(label)):
        countSet.append(0.0)
        predictSet.append(0.0)
        rightSet.append(0.0)

    for p, r in zip(predict, real):
        if(isinstance(p,list)):
            indexP = np.argmax(p)
        else:
            indexP = p
        if(isinstance(r,list)):
            indexR = np.argmax(r)
        else:
            indexR = r
        # name = label[int(indexR)]
        countSet[int(indexR)] += 1
        predictSet[int(indexP)] += 1
        if (int(indexR) == int(indexP)):
            rightSet[int(indexP)] += 1
    count = 0

    table = PrettyTable(["Tag name", "count", "precision", "recall"])
    table.align["Tag name"] = "l"
    table.padding_width = 1
    result = ''
    for i, j, k in zip(countSet, predictSet, rightSet):
        tag = str(label[count])
        if (i == 0):
            recall = '0'
        else:
            recall = str(round(k / i,3))
        if (j == 0):
            precision = '0'
        else:
            precision = str(round(k / j,3))
        temp = tag + ',' + str(i) + ',' + str(precision) + ',' + str(recall)
        result+=temp+'\n'
        table.add_row([tag ,str(i),str(precision),str(recall)])
        # result_file.write(tag+','+ str(i) +','+str(precision)+','+str(recall)+'\n')
        count += 1
    table_string = table.get_string()
    print table_string
    return result