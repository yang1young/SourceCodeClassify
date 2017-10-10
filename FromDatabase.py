import codecs
import mysql.connector
import os
import CleanUtils as cc


tagSet = ['opengl','sockets','sorting', 'mfc','lambda', 'random','math','io', 'openmp','xcode',
          'arduino','jni','mingw','tree','directx','time', 'openssl','network','hash','mysql',
          'heap', 'gtk', 'graph']
path = '/home/qiaoyang/bisheData/'


def codeIntoSepFile():
    conn = mysql.connector.connect(user='root', password='1qazxc', database='codetag')  # , use_unicode=True
    cursor = conn.cursor(buffered=True)
    cursor.execute('select * from selectTag')
    numRows = int(cursor.rowcount)
    #tagFile = open('/home/qiaoyang/pythonProject/jeornTest/selectTag/tag.txt', 'w+')

    for i in range(numRows):
        row = cursor.fetchone()
        id = str(row[0])
        code = row[1]
        tag = str(row[2])
        f =  codecs.open(path+'codeData/'+str(i)+"-"+tag+'.c','w+','utf8')
        f.write(code)
        #tagFile.write(tag)
        f.close()
        print i
    #tagFile.close()


def select_code_data():
    conn = mysql.connector.connect(user='root', password='1qazxc', database='codetag')  # , use_unicode=True
    cursor = conn.cursor(buffered=True)
    cursorInsert = conn.cursor()
    cursor.execute('select * from SampleC')
    numRows = int(cursor.rowcount)
    print numRows

    for i in range(numRows):
        row = cursor.fetchone()
        id = row[0]
        code = cc.code_clean(cc.remove_cpp_comment(row[1].encode('utf-8')))
        tags = str(row[2]).split('#')
        finalTag = ''

        for tag in tags:
            if (tag in tagSet):
                finalTag =tag

        if (finalTag != ''):
            sql = "insert into selectTag(Id,Code,Tags) values(%s,%s,%s)"
            data = [id, code, finalTag]
            # print data
            cursorInsert.execute(sql, data)
        print str(i)+"----"+str(numRows)
    conn.commit()
    conn.close()


def type_mix_code():

    pathNew = path+"astData/"
    pathOrigin = path+"codeData/"
    originFiles = os.listdir(pathOrigin)
    originFiles.sort(key=lambda x: int(str(x).split("-")[0]))
    #print originFiles

    newFiles = os.listdir(pathNew)
    newFiles.sort(key=lambda x: int(str(x).split("-")[0]))
    #print newFiles.__len__()


    # tagCountDict = dict()
    # for tags in tagLines:
    #     tag = str(tags).split(' ')
    #     for t in tag:
    #         if(tagCountDict.__contains__(t)):
    #             tagCountDict[t] +=1
    #         else:
    #             tagCountDict[t] = 1
    conn = mysql.connector.connect(user='root', password='1qazxc', database='codetag')  # , use_unicode=True
    cursorInsert = conn.cursor()
    count = 1
    for file in newFiles:

        f1 = codecs.open(pathNew + file, 'r', 'utf8')
        type = f1.read().replace('\n',' ')+'\n'#cc.cleanRegex(f.read())
        f1.close()
        f2 = codecs.open(pathOrigin + file, 'r', 'utf8')
        code = f2.read().replace('\n', ' ') + '\n'  # cc.cleanRegex(f.read())
        f2.close()
        tag = file.split(".")[0].split("-")[1]

        sql = "insert into selectTagType(Id,Code,Type,Tags) values(%s,%s,%s,%s)"
        data = [int(str(file).split("-")[0]),code,type, tag]
        #cursorInsert.execute(sql, data)
        try:
            cursorInsert.execute(sql, data)
        except Exception:
            print file
        count +=1
        print count

    conn.commit()
    conn.close()



#select_code_data()
#codeIntoSepFile()
type_mix_code()