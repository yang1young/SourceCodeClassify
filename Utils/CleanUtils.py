import re
import codecs
import mysql.connector

SPLIT_CHARS = [',','+','&','!','%','?','_','|',':','-','=','\\','~','*','^','<','>','[',']','$','{','}',';','.','`','@','(',')']
_WORD_SPLIT = re.compile(b"([,+\-&!%'_?|=\s/\*^<>$@\[\](){}#;])")
KEY_WORD_PATH = '/home/qiaoyang/bishe/SourceCodeClassify/keyword.txt'


# all kind of split char
def get_split_set():
    split_set = set()
    for chars in SPLIT_CHARS:
        split_set.add(chars)
    return  split_set

def remove_non_ascii_1(text):
    return ''.join([i if ord(i) < 128 else ' ' for i in text])


def remove_cpp_comment(text):
    def blotOutNonNewlines(strIn):  # Return a string containing only the newline chars contained in strIn
        return "" + ("\n" * strIn.count('\n'))

    def replacer(match):
        s = match.group(0)
        if s.startswith('/'):  # Matched string is //...EOL or /*...*/  ==> Blot out all non-newline chars
            return blotOutNonNewlines(s)
        else:  # Matched string is '...' or "..."  ==> Keep unchanged
            return s
    pattern = re.compile(
        r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
        re.DOTALL | re.MULTILINE
    )
    return re.sub(pattern, replacer, text)


def code_clean(text):
    patternBlank = re.compile(' +')
    patternDouble2 = re.compile('\\n\s*\\n')
    patternPrintf = re.compile('.*?print.*\r?\n')
    patternCout = re.compile('.*?cout.*\r?\n')
    patternCin = re.compile('.*?cin.*\r?\n')
    patternScanf = re.compile('.*?scanf.*\r?\n')
    patternReader = re.compile('.*?Reader.*\r?\n')
    patternStream = re.compile('.*?Stream.*\r?\n')
    patternWriter = re.compile('.*?Writer.*\r?\n')
    patternInclude = re.compile('.*?include.*\r?\n')
    patternNameSpace = re.compile('.*?namespace.*\r?\n')


    a = re.sub(patternPrintf, "", text)
    a = re.sub(patternCout, "", a)
    a = re.sub(patternCin, "", a)
    a = re.sub(patternScanf, "", a)
    a = re.sub(patternReader, "", a)
    a = re.sub(patternStream, "", a)
    a = re.sub(patternWriter, "", a)
    a = re.sub(patternInclude, "", a)
    a = re.sub(patternNameSpace, "", a)
    a = re.sub(patternBlank, " ", a)
    a = re.sub(patternDouble2, "\n", a)

    return a


def code_split(code, isSplit):

    if(isSplit):
        numA = code.count("\n")
        # print numA
        codes = code.split("\n")
        result = ""
        count = 0
        numBlock = 1
        for item in codes:
            count += 1
            result = result + item + " "
            # if(count!=len(codes)&count!=len(codes)-1):
            # result = result +"$"
            if ((count % 7 == 0) & (numA - count > 7)):
                result = result + '\n'
                numBlock += 1
        # print (result+"\n", numBlock)
        return (result + "\n", numBlock)
    else:
        return (code.replace("\n", " ") + "\n", 1)


def get_type_set():
    f = codecs.open(KEY_WORD_PATH, 'r', 'utf8')
    lines = f.readlines()
    tagSet = set()
    for line in lines:
        tagSet.add(line.encode('utf-8').replace('\n', ''))
    return tagSet


# make code anonymous, such as all number replaced by NUMBER
# all string replaced by STRING, all variable changed to VAR,etc
def code_anonymous(code):
    keyword = get_type_set()

    # repalce string
    patterString = re.compile("\"(.*?)\"")
    code = re.sub(patterString, "STRING", code)

    # split by slicers
    codes = _WORD_SPLIT.split(code)

    final_code = ''
    for code in codes:
        if ((code == ' ') | (code in keyword)):
            final_code += code
        elif (code != ''):
            # replace number
            if (code.isdigit()):
                final_code += 'NUMBER'
            # # if the variable or function's name is long, we keep it
            # elif (code.__len__() >= 3):
            #     final_code += code
            # # if the variable or function's name is short, we replace it
            # elif (code.__len__() < 3):
            #     final_code += ' VAR '
            else:
                final_code += ' VAR '
    return final_code


def string_reverse(string):
    lst = string.split()  # split by blank space by default
    return ' '.join(lst[::-1])


# make sure every split char is blank
def get_normalize_code(code,max_lenghth):
    if(code is None):
        return ""
    split_set = get_split_set()
    codes= _WORD_SPLIT.split(code)
    result = ''
    count_length = 0
    for c in codes:
        if (c != ''):
            if (c in split_set):
                result += ' '+c+' '
            else:
                result += c
            count_length += 1
        if (count_length == max_lenghth):
            break
    result = " ".join(result.split())
    return result



# order AST type
def AST_type_clean(line_dict, need_repeated):
    line_code = []
    newDict = sorted(line_dict.iteritems(), key=lambda d: d[0])

    for key, value in newDict:
        if (need_repeated):
            remove_duplicated = sorted(str(value).split(' '))
        else:
            remove_duplicated = sorted(set(str(value).split(' ')))
        line_code.append(' '.join(e for e in remove_duplicated))

    return ','.join(e for e in line_code)


def remove_blank(texts):
    new_texts = []
    for text in texts:
        if(text!='' and text!=' '):
            new_texts.append(text)
    return new_texts