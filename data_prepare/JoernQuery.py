from joern.all import JoernSteps
import codecs
from py2neo.packages.httpstream import http
http.socket_timeout = 9999


path = '/home/qiaoyang/bisheData/astData/'

def addInfoToSourceFile(text,filePath):
    if(text!=''):
        f = codecs.open(path+str(filePath), 'a', 'utf8')
        f.write(text+'\n')
        f.close()


def getCleanText(lineDict,doRemove):
    lineText = []

    newDict = sorted(lineDict.iteritems(), key=lambda d: d[0])

    for key,value in newDict:
        if (doRemove):
            removeDuplicated = sorted(set(str(value).split(' ')))
        else:
            removeDuplicated = sorted(str(value).split(' '))
        lineText.append(' '.join(e for e in removeDuplicated))

    return  ','.join(e for e in lineText)


def runQuery():

    j = JoernSteps()
    j.setGraphDbURL('http://localhost:7474/db/data/')
    j.connectToDatabase()

    query = """getNodesWithType('Function')"""
    res=j.runGremlinQuery(query)
    flag = 1
    for function in res:
        if (flag):
            lineDict = dict()
            functionnodeid = int(function.ref[5:])
            #query = """g.v(%d).functionToAST().astNodes()""" % (functionnodeid)
            #allNodesOfFunction1 = j.runGremlinQuery(query)

            query = """queryNodeIndex("functionId:%i").as("x").statements().as("y").select{it.type}{it.location}"""%functionnodeid
            allNodesOfFunction = j.runGremlinQuery(query)

            for node in allNodesOfFunction:
                #print node
                type = str(node[0])
                location = str(node[1])
                if(location!='None'):
                    loc = str(location).split(':')[0]
                    if (lineDict.has_key(loc)):
                        temp = lineDict.get(loc) + ' ' + type
                        lineDict[loc] = temp
                    else:
                        lineDict[loc] = type


            text = getCleanText(lineDict,False)
            #print text
            query = """g.v(%d).in("IS_FILE_OF").filter{it.type=="File"}.filepath""" % functionnodeid
            filepath = j.runGremlinQuery(query)
            fileName = str(filepath[0]).split('/')[-1]
            addInfoToSourceFile(text,fileName)
        flag +=1
        print flag


def runQueryChunk():

    j = JoernSteps()
    j.setGraphDbURL('http://localhost:7474/db/data/')
    j.connectToDatabase()

    query = """getNodesWithType('Function').id"""
    res=j.runGremlinQuery(query)
    flag = 1
    CHUNK_SIZE=51

    for chunk in j.chunks(res,CHUNK_SIZE):
        if(flag):
            functionTuple = tuple(chunk)
            functionIdStr = str(functionTuple)
            functionIdStr=functionIdStr.replace(',','')
            functionIdStr=functionIdStr.replace('\'','')

            #query = """queryNodeIndex("functionId:%s").as("x").statements().map("functionId","location").as("y").select{it.type}{it}""" % functionIdStr
            query = """queryNodeIndex("functionId:%s").as("x").statements().as("y").as("z").select{it.type}{it.location}{it.functionId}""" % functionIdStr
            stms = j.runGremlinQuery(query)

            query = """idListToNodes(%s).as("x").in("IS_FILE_OF").filepath.as("y").select{it.id}{it}""" % chunk
            stmsFiles = j.runGremlinQuery(query)
            files = dict()
            for stmsFile in stmsFiles:
                files[int(stmsFile[0])] = str(stmsFile[1]).split('/')[-1]

            codes = dict()
            for stm in stms:
                functionnodeid = int(stm[2])
                loc = stm[1]
                type =str(stm[0])

                if(codes.__contains__(functionnodeid)):
                    codes[functionnodeid].append([loc,type])
                else:
                    codeList = [[loc,type]]
                    codes[functionnodeid] = codeList

            codesList = codes.items()
            for id,elem in codesList:
                lineDict = dict()
                for e in elem:
                    location = str(e[0])
                    type = e[1]

                    if (location != u'None'):
                        loc = str(location).split(':')[0]
                        if (lineDict.has_key(loc)):
                            temp = lineDict.get(loc) + ' ' + type
                            lineDict[loc] = temp
                        else:
                            lineDict[loc] = type
                text = getCleanText(lineDict, False)
                fileName = files.get(id)
                addInfoToSourceFile(text, fileName)
            flag += 1
            print flag

runQueryChunk()

''' if(node.properties[u'isCFGNode']):
                  type = str(node.properties[u'type'])
                  location = node.properties[u'location']
                  loc = str(location).split(':')[0]
                  if(lineDict.has_key(loc)):
                      temp = lineDict.get(loc)+' '+type
                      lineDict[loc] = temp
                  else:
                      lineDict[loc] = type
             '''