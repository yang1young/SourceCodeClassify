import codecs
import os
import re
from Utils import CleanUtils as cc


path = '/home/qiaoyang/codeData/molili/'
code_path = path+'sepCodeFile/'
ast_path = path+'sepTypeFile/'

write_to = '/home/qiaoyang/bishe/SourceCodeClassify/data_molili/'
tagSet = range(0,104)


def prepare_csv():
    code_files = os.listdir(code_path)
    print len(code_files)

    ast_files = os.listdir(ast_path)
    print len(ast_files)
    ast_files.sort(key=lambda x: x[:-2])
    print ''

    code_train = codecs.open(write_to + 'train.txt', 'w+', 'utf8')
    code_dev = codecs.open(write_to + 'dev.txt', 'w+', 'utf8')
    code_test = codecs.open(write_to + 'test.txt', 'w+', 'utf8')

    for file in ast_files:
        print file
        type = open(ast_path+file,'r').read()
        code = open(code_path+file,'r').read()

        code = cc.code_anonymous(
            cc.get_normalize_code(cc.remove_non_ascii_1(code.encode('utf-8')).replace("\n", " "), 1000))

        patternBlank = re.compile(' +')
        code = re.sub(patternBlank, " ", code).replace("@", "")
        type = cc.string_reverse(type.replace('\n', ''))

        tag = int(file.split(".")[0].split("-")[0])-1
        i = int(file.split(".")[0].split("-")[1])
        print i
        if ( i< 500* 0.7):
            code_train.write(str(tag) + "@" + code + "@" + type + "\n")
        elif (i < 500* 0.8):
            code_dev.write(str(tag) + "@" + code + "@" + type + "\n")
        else:
            code_test.write(str(tag) + "@" + code + "@" + type + "\n")


#prepare_csv()