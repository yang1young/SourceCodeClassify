import codecs
import mysql.connector
import os
import CleanUtils as cc


tagSet = ['opengl','sockets','sorting', 'mfc','lambda', 'random','math','io', 'openmp','xcode',
          'arduino','jni','mingw','tree','directx','time', 'openssl','network','hash','mysql',
          'heap', 'gtk', 'graph']


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
        f =  codecs.open('/home/qiaoyang/bisheData/codeData/'+str(i)+"-"+tag+'.c','w+','utf8')
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

        # if(finalTag!=""):
        #     print id
        #     print code
        #     print finalTag
        if (finalTag != ''):
            sql = "insert into selectTag(Id,Code,Tags) values(%s,%s,%s)"
            data = [id, code, finalTag]
            # print data
            cursorInsert.execute(sql, data)
        print str(i)+"----"+str(numRows)
    conn.commit()
    conn.close()

#select_code_data()
codeIntoSepFile()
