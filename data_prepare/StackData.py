import codecs
import os
import re
import mysql.connector
from Utils import CleanUtils as cc


path = '/home/qiaoyang/bishe/SourceCodeClassify/data_so/'
sep_path = '/home/qiaoyang/bisheData/'
# tagSet = ['opengl','sockets','sorting', 'mfc','lambda', 'random','math','io', 'openmp','xcode',
#           'arduino','jni','mingw','tree','directx','time', 'openssl','network','hash','mysql',
#           'heap', 'gtk', 'graph']

tagSet = ['opencv','opengl','file','sockets','memory','matrix','sorting','recursion','optimization','assembly',
          'mfc','lambda','macros','casting','random','osx','math','polymorphism','parsing','arduino','iterator',
          'assembly','jni','mingw','tree','directx','time', 'openssl','hash','mysql']

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
        f =  codecs.open("/home/qiaoyang/bisheData/codeData/"+str(i)+"-"+tag+'.c','w+','utf8')
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
    data_dict = dict()
    for i in range(numRows):
        row = cursor.fetchone()
        id = row[0]
        code = cc.code_clean(cc.remove_cpp_comment(row[1].encode('utf-8')))
        tags = str(row[2]).split('#')
        finalTag = ''

        for tag in tags:
            if (tag in tagSet):
                if(tag in data_dict):
                    data_dict[tag] +=1
                else:
                    data_dict[tag] = 1
                finalTag =tag

        if (finalTag != ''):
            sql = "insert into selectTag(Id,Code,Tags) values(%s,%s,%s)"
            data = [id, code, finalTag]
            # print data
            cursorInsert.execute(sql, data)
        print str(i)+"----"+str(numRows)
    for k, v in data_dict.iteritems():
        print k, v
    conn.commit()
    conn.close()


def type_mix_code():

    pathNew = sep_path+"astData/"
    pathOrigin = sep_path+"codeData/"
    #originFiles = os.listdir(pathOrigin)
    #originFiles.sort(key=lambda x: int(str(x).split("-")[0]))
    #print originFiles

    newFiles = os.listdir(pathNew)
    newFiles.sort(key=lambda x: (str(x).split("-")[1],int(str(x).split("-")[0])))
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



def create_dict():
    tagDict = dict()
    i =0
    for tag in tagSet:
        tagDict[tag]=i
        i+=1
    return tagDict

def prepare_data():
    conn = mysql.connector.connect(user='root', password='1qazxc', database='codetag')  # , use_unicode=True
    cursor = conn.cursor(buffered=True)
    cursor.execute('select * from selectTagType')
    numRows = int(cursor.rowcount)

    code_train = codecs.open(path+'trainCode.txt', 'w+', 'utf8')
    type_train = codecs.open(path+'trainType.txt', 'w+', 'utf8')
    tag_train = codecs.open(path +'trainTag.txt', 'w+', 'utf8')

    code_test = codecs.open(path + 'testCode.txt', 'w+', 'utf8')
    type_test = codecs.open(path + 'testType.txt', 'w+', 'utf8')
    tag_test = codecs.open(path + 'testTag.txt', 'w+', 'utf8')

    for i in range(numRows):
        row = cursor.fetchone()
        id = row[0]
        code = cc.code_anonymous(cc.get_normalize_code(cc.remove_non_ascii_1(row[1].encode('utf-8')).replace("\n"," "),1000))
        patternBlank = re.compile(' +')
        code = re.sub(patternBlank, " ", code)
        type = str(row[2]).replace('\n','')
        tag = str(row[3]).replace('\n','')
        if(i<numRows*0.8):
            code_train.write(code+'\n')
            type_train.write(type+'\n')
            tag_train.write(tag+'\n')
        else:
            code_test.write(code + '\n')
            type_test.write(type + '\n')
            tag_test.write(tag + '\n')
    code_train.close()
    type_train.close()
    tag_train.close()
    code_test.close()
    type_test.close()
    tag_test.close()


def prepare_csv():
    conn = mysql.connector.connect(user='root', password='1qazxc', database='codetag')  # , use_unicode=True
    cursor = conn.cursor(buffered=True)
    tagDict = create_dict()

    code_train = codecs.open(path+'train.txt', 'w+', 'utf8')
    code_dev = codecs.open(path + 'dev.txt', 'w+', 'utf8')
    code_test = codecs.open(path + 'test.txt', 'w+', 'utf8')

    for t in tagSet:
        #cursor.execute('select * from selectTagType where ')
        cursor.execute('select * from selectTagType where Tags = %s ORDER BY Id',(t,))
        numRows = int(cursor.rowcount)
        print str(numRows)+'---------'+t
        for i in range(numRows):
            row = cursor.fetchone()
            id = row[0]
            code = cc.code_anonymous(
                cc.get_normalize_code(cc.remove_non_ascii_1(row[1].encode('utf-8')).replace("\n", " "), 1000)).replace('\x00', '')
            #code = cc.get_normalize_code(cc.remove_non_ascii_1(row[1].encode('utf-8')).replace("\n", " "), 1000).replace("@","").replace('\x00', '')
            patternBlank = re.compile(' +')
            code = re.sub(patternBlank, " ", code).replace("@", "")
            type = cc.remove_dupliacte(cc.string_reverse(str(row[2]).replace('\n', '')))
            tag = str(tagDict.get(str(row[3]).replace('\n', '')))
            if (i < numRows * 0.7):
                code_train.write(tag + "@" + code + "@" + type + "\n")
            elif (i < numRows * 0.8):
                code_dev.write(tag + "@" + code + "@" + type + "\n")
            else:
                code_test.write(tag + "@" + code + "@" + type + "\n")
            #print str(id)+"---"+t

    code_train.close()
    code_dev.close()
    code_test.close()


#select_code_data()
#codeIntoSepFile()
#type_mix_code()
#prepare_csv()