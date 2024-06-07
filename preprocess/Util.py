from os.path import join, exists, basename, dirname, abspath, splitext, abspath
from subprocess import run
import csv, argparse
import json, re
import os, sys, multiprocessing
from os.path import join

class CSVFile():
    """docstring for CSVFile"""
    def __init__(self, arg, csvfile):
        self.arg = arg
        self.csv_file = csvfile

        with open(self.csv_file, 'r', newline='') as csvfile: 
            csv_reader = csv.reader(x.replace('\0', '') for x in csvfile) 

            # extracting field names through first row 
            head = next(csv_reader) 
            self.col_year = head.index("year")
            self.col_flines = head.index("flines")
            self.col_file = head.index("file")
            self.col_author = head.index("username")
            self.col_fullpath = head.index("full_path")
            self.col_solution = head.index("solution")
            self.col_task = head.index("task")
            self.col_round = head.index("round")

        self.str_num = 'num'
        self.str_content = 'src'
        csv.field_size_limit(sys.maxsize)

    def getAllCodeByNum(self, num):
        self.souce_code_dir = self.arg.srcDir
        if self.souce_code_dir != None:
            if not exists(self.souce_code_dir):
                os.makedirs(self.souce_code_dir)

        codeWithAuthDict = self.collectSrcCode()
        authorSrcInfo = { author: value for (author, value) in codeWithAuthDict.items() 
                        if value[self.str_num] >= num }
        if len(authorSrcInfo.keys()) == 0:
            pass
        else:
            with open(self.csv_file, 'r', newline='') as csvfile: 
                # creating a csv reader object
                csv_reader = csv.reader(x.replace('\0', '') for x in csvfile)

                # extracting field names through first row 
                fields = next(csv_reader) 

                # extracting each data row one by one 
                for row in csv_reader: 
                    if '/' in row[self.col_author]:
                        continue
                    author = row[self.col_year]+"_"+row[self.col_author]
                    # print(row[self.col_author])
                    fullpath = row[self.col_fullpath] if row[self.col_fullpath] != "" else row[self.col_file]
                    if author in authorSrcInfo.keys():
                        if fullpath in authorSrcInfo[author][self.str_content]:
                            output_name = author+"_"+fullpath
                            filename = join(self.souce_code_dir, output_name)
                            if exists(filename):
                                continue
                            source_code_f = open(filename, 'w+')
                            source_code_f.write(row[self.col_flines])
                            source_code_f.close()

        print("there are {} authors that meet requirements".format(len(authorSrcInfo.keys())))
        return authorSrcInfo

    def collectSrcCode(self):
        postfix = ['c', 'c++', 'cpp']
        with open(self.csv_file, 'r', newline='') as csvfile: 
            # creating a csv reader object
            csv_reader = csv.reader(x.replace('\0', '') for x in csvfile)

            # extracting field names through first row 
            fields = next(csv_reader) 

            dictData = dict()
            # extracting each data row one by one 
            for row in csv_reader: 
                filename = row[self.col_fullpath] if row[self.col_fullpath] != "" else row[self.col_file]
                if len(filename.split('.'))<2:
                    continue
                if filename.split('.')[1].lower() in postfix:
                    author = row[self.col_year]+"_"+row[self.col_author]
                    if author in dictData.keys():
                        dictData[author].append(filename)
                    else:
                        dictData[author] = [filename]

        codeWithAuthDict = { author:{self.str_num:len(dictData[author]),self.str_content:dictData[author]} 
                            for author in dictData.keys() }
        return codeWithAuthDict

    def getStatistic4C(self):
        """ collect all c code in this csv file """
        postfix = '.c'
        # {year:{'authorx': [], ...}}
        dictData = dict()
        dictData[self.col_year] = dict()
        with open(self.csv_file, 'r', newline='') as csvfile:
            # creating a csv reader obj
            csv_reader = csv.reader(x.replace('\0', '') for x in csvfile)

            # extracting field names through 1st row
            fields = next(csv_reader)

            # extracting each data row one by one
            for row in csv_reader:
                filename = row[self.col_fullpath] if row[self.col_fullpath] != '' else row[self.col_file]
                _, extension = splitext(basename(filename))
                if extension.lower() != postfix:
                    continue
                else:
                    if row[self.col_author] in dictData[self.col_year].keys(): 
                        dictData[self.col_year][row[self.col_author]].append(filename)
                    else:
                        dictData[self.col_year][row[self.col_author]] = [filename]
        return dictData

    def getAllCodeByAuthors(self, authors):
        self.souce_code_dir = self.arg.srcDir
        if self.souce_code_dir != None:
            if not exists(self.souce_code_dir):
                os.makedirs(self.souce_code_dir)

        srcAuthorsInfo = self.getSrcByAuthors(authors)
        for author in authors:
            for src in srcAuthorsInfo[author][self.str_content]:
                if exists(filename):
                    continue
                source_code_f = open(filename, 'w+')
                source_code_f.write(src)
                source_code_f.close()

    def getSrcByAuthors(self, authors):
        postfix = ['c', 'c++', 'cpp']
        with open(self.csv_file, 'r', newline='') as csvfile: 
            # creating a csv reader object
            csv_reader = csv.reader(x.replace('\0', '') for x in csvfile)

            # extracting field names through first row 
            fields = next(csv_reader) 

            dictData = dict()
            # extracting each data row one by one 
            for row in csv_reader: 
                if row[self.col_author] not in authors:
                    continue
                else:
                    # filename = row[self.col_fullpath] if row[self.col_fullpath] != "" else row[self.col_file]
                    filename = "_".join([row[self.col_round], row[self.col_task], row[self.col_solution], row[self.col_author]])
                    if len(filename.split('.'))<2:
                        continue
                    if filename.split('.')[1].lower() in postfix:
                        author = row[self.col_year]+"_"+row[self.col_author]
                        if author in dictData.keys():
                            dictData[author].append(filename)
                        else:
                            dictData[author] = {self.str_name: filename, self.str_code: row[self.col_flines]}

                    codeWithAuthDict = { author:{self.str_num:len(dictData[author]),self.str_content:dictData[author]} 
                        for author in dictData.keys() }
        return codeWithAuthDict

class BinFiles():
    """docstring for BinFile"""
    def __init__(self, arg):
        self.arg = arg
        cmd_find = 'find ' + arg.binDir+' -executable -type f'
        self.binFiles = run(cmd_find, shell=True, capture_output=True).stdout.decode('utf-8').split('\n')[:-1]
        self.binInfoJson = join(arg.binDir, "bin_Info.json")
        if not exists(self.binInfoJson):
            self.buildBinInfoJson()
        else:
            with open(self.binInfoJson, 'r') as f:
                self.binInfo = json.load(f)

        self.dbRoot = arg.dbDir

    def buildBinInfoJson(self):
        binInfo = dict()
        # bin name containing author name
        for binary in self.binFiles:
            author = self.getAuthorName(binary)
            if author in binInfo.keys():
                binInfo[author].append(binary)
            else:
                binInfo[author] = [binary]

        with open(self.arg.binInfoJson, 'w') as f:
            json.dump(binInfo, f, indent=4)

        self.binInfo = binInfo

    def getAuthorName(self, binary):
        author = basename(binary).split("_")[2:]
        return author

    def updateJsonFile(self, file, data):
        # data's value must be in list form
        if exists(file):
            with open(file, 'r') as f:
                oldInfo = json.load(f)
        else:
            oldInfo = dict()
        # combine binInfoJson with old one 
        for key in oldInfo.keys():
            if key not in data.keys():
                data[key] = oldInfo[key]
            else:
                data[key] = list(set(data[key]+oldInfo[key]))
        # save the latest info
        with open(file, 'w') as f:
            json.dump(data, f, indent=4)

    def buildDB(self):
        pass
        

class SrcFiles():
    """docstring for SrcFile"""
    def __init__(self, arg, srcFiles=None):
        self.arg = arg
        self.srcDirPath = arg.srcDir
        if srcFiles == None:
            cmd = 'find '+arg.srcDir+' -type f -iname "*.c" -o -iname "*.cpp"'
            proc = run(cmd, shell=True, capture_output=True)
            self.srcFiles = proc.stdout.decode('utf-8').split('\n')[:-1]
        else:
            self.srcFiles = [join(self.srcDirPath, src) for src in srcFile]
        print("len of src is {}".format(len(self.srcFiles)))
        self.binDirPath = arg.binDir
        if not exists(self.binDirPath):
            run('mkdir -p '+self.binDirPath, shell=True)
        self.binInfoJson = join(self.binDirPath, "bin_Info.json")
        if not exists(self.binInfoJson):
            self.binInfo = dict()
        else:
            with open(self.binInfoJson, 'r') as f:
                self.binInfo = json.load(f)
        self.optLvs = ['-O0', '-O1', '-O2', '-O3']
        self.ldOpts = " -lm -lc -lpthread -lstdc++ -ldl "

    def chooseFile(self, cond):
        pass

    def compile(self):
        # compile src for diff opt levels. bin names end with '-optX'
        if not self.arg.mixed:
            optLvs = self.optLvs[:1]
        else:
            optLvs = self.optLvs
        for optLv in optLvs:
            # parallelly run compilation
            arguments = [(optLv, src) for src in self.srcFiles]
            pool = multiprocessing.Pool()
            compile_rslts = pool.starmap(self.compileSrcFile, arguments)
            pool.close()
            pool.join()

            # recording authors and corresponding binaries paths
            for e in compile_rslts:
                if e[1] == None:
                    # compilation failed
                    continue
                else:
                    src, bin_path = e
                    author = self.getAuthor(src)
                    if author not in self.binInfo.keys():
                        self.binInfo[author] = [bin_path]
                    else:
                        # self.binInfo may read from a JSON file
                        if bin_path not in self.binInfo[author]:
                            self.binInfo[author].append(bin_path)

        # save all authors and corresponding binaries info to a json file
        # read the json file first if exists
        if exists(self.binInfoJson):
            with open(self.binInfoJson, 'r') as f:
                oldBinInfo = json.load(f)
        else:
            oldBinInfo = dict()
        # combine binInfoJson with old one 
        for author in oldBinInfo.keys():
            if author not in oldBinInfo.keys():
                self.binInfo[author] = oldBinInfo[author]
            else:
                self.binInfo[author] = list(set(self.binInfo[author]+oldBinInfo[author]))
        # save the latest info
        with open(self.binInfoJson, 'w') as binInfoJson:
            json.dump(self.binInfo, binInfoJson, indent=4)
        print("saved all authors and corresponding binaries into {}".format(self.binInfoJson))

    def getAuthor(self, src):
        file_name = splitext(basename(src))[0]
        author = "_".join(file_name.split("_")[:-1])
        return author

    def compileSrcFile(self, optLv, src):
        lv = optLv[-1]
        if src.lower().endswith('c'):
            compiler = 'gcc'
        else:
            compiler = 'g++'

        # binary_name includes authors' name, end with '-optX'
        binary_name = splitext(basename(src))[0]+'-opt'+lv
        bin_path = join(abspath(self.binDirPath), binary_name)
        if exists(bin_path):
            return src, bin_path
        cmd = compiler + " -m32 " + optLv + " " + src + " -o " + bin_path + self.ldOpts
        p = run(cmd, capture_output=True, shell=True)
        if p.returncode != 0:
            with open("compile_err.log", 'a') as f:
                f.write("working for {}\ncmd : {}\n".format(src, cmd))
                f.write(p.stderr.decode('utf-8')+'\n')
            return src, None
        else:
            # record new binary path to the db json file
            return src, bin_path

    def getSrcInfo(self):
        pass

def get_names_with_star(nameParts, start):
    fileNameParts = nameParts
    try:
        l_fn_name = int(fileNameParts[start])
        start += 1
        fn_name = "_".join(fileNameParts[start:(start+l_fn_name)])

        start = start+l_fn_name
        l_author = int(fileNameParts[start])
        start += 1
        author = "_".join(fileNameParts[start:(start+l_author)])

        start += l_author
        l_bin = int(fileNameParts[start])
        start += 1
        binary_name = "_".join(fileNameParts[start:(start+l_bin)])
        x = [str(l_fn_name), fn_name, str(l_author), author, str(l_bin), binary_name]
        return fn_name, author, binary_name, "_".join(x)
    except Exception as e:
        print(nameParts, start)
        raise e

def get_fname_author_bin_from_parts(fileNameParts):
    # slice file name contain: 'rest'/(api, argInd,), fAddr, fname, author, binary name;
    # function name, author, and binary name are in the form of 'L+V'
    start = 0
    # fAddr start
    if fileNameParts[0] == 'rest':
        start += 1
    else:
        start += 2
    # fname start
    start += 1
    try:
        return get_names_with_star(fileNameParts, start)
    except Exception as e:
        print(fileNameParts, start)
        raise e

def get_names_from_normal_parts(normNameParts):
    # normal name contain: fname, author, binary name;
    # function name, author, and binary name are in the form of 'L+V'
    start = 0

    return get_names_with_star(normNameParts, start)

def get_Author_num_from_dir(directory):
    fns = []
    authors = []
    author_fns_dict = dict()
    cmd = "find "+directory+" -mindepth 1 -maxdepth 1 -type d "
    process = run(cmd, shell=True, capture_output=True)
    subdirs = process.stdout.decode('utf-8').strip().split("\n")
    print("there are {} sub-dir".format(len(subdirs)))
    for subdir in subdirs:
        # slice file name contain: 'rest'/(api, argInd,), fAddr, fname, author, binary name;
        # function name, author, and binary name are in the form of 'L+V'
        fileNameParts = basename(subdir).split('_')
        fn_name, author, _, _ = get_fname_author_bin_from_parts(fileNameParts)
        if author not in author_fn_dict.keys():
            author_fn_dict[author] = [fn_name]
        else:
            author_fn_dict[author] += [fn_name]
        # fns.append(fn_name)
        # authors.append(author)

    # authors = list(set(authors))
    # fns = list(set(fns))
    for author in author_fn_dict.keys():
        no_repetation = list(set(author_fn_dict[author]))
        author_fn_dict[author] = no_repetation
    sorted_author_fn_dict = dict(sorted(author_fn_dict.items(), key=lambda item: item[1]))
    print("there are {} authors;\tthere are {} fns".format(len(author_fn_dict.keys()), len(fns)))
    print(authors)

def get_info_from_slices(slices_list):
    print(sys._getframe().f_lineno)
    info = dict()
    for s in slices_list:
        slice_name_parts = basename(s).split("_")
        fn_name, author, binary_name = get_fname_author_bin_from_parts(slice_name_parts)
        if binary_name not in info.keys():
            info[binary_name] = dict()

        if author not in info[binary_name].keys():
            info[binary_name][author] = []

        info[binary_name][author].append(fn_name)

    print(sys._getframe().f_lineno)
    info = json.dumps(info, indent=4)
    print(info)

def get_fn_from_dir_by_authors(directory, authors):
    fns = dict()
    for author in authors:
        fns[author] = []
    cmd = "find "+directory+" -mindepth 1 -maxdepth 1 -type d "
    process = run(cmd, shell=True, capture_output=True)
    subdirs = process.stdout.decode('utf-8').strip().split("\n")
    print("there are {} sub-dir".format(len(subdirs)))
    for subdir in subdirs:
        # slice file name contain: 'rest'/(api, argInd,), fAddr, fname, author, binary name;
        # function name, author, and binary name are in the form of 'L+V'
        fileNameParts = basename(subdir).split('_')
        fn_name, author, _ = get_fname_author_bin_from_parts(fileNameParts)
        if author in authors:
            fns[author].append(fn_name)
                              
    for author in authors:
        fns[author] = len(set(fns[author]))
         
    print(fns)

def get_num_records(directory):
    records = []
    cmd = "find "+directory+" -type f -iname '*.log'"
    process = run(cmd, shell=True, capture_output=True)
    files = process.stdout.decode('utf-8').strip().split("\n")

    seperate_ptn = re.compile(r'(\*\*==)+')
    angr_ptn = re.compile(r'angr identified \d+ funs')
    for file in files:
        # print(file)
        with open(file, 'r') as f :
            lines = f.readlines()

        num_in_record = 0
        cnt = 0
        for line in lines:
            if angr_ptn.search(line):
                num_in_record = int(line.strip().split(" ")[2])
            if seperate_ptn.search(line):
                cnt += 1
            else:
                continue

        if num_in_record == cnt:
            pass
        else:
            print("{} record {} funs but processed {} funs".format(file, num_in_record, cnt))

def get_stat_info_from_dir(directory):
    stat_info_file = join(directory, 'repo_statistic_info.txt')
    if exists(stat_info_file):
        run("rm "+stat_info_file, shell=True)
    
    cmd = "find "+join(directory, 'repos')+" -type f -iname 'final_result_ratio.txt'"
    print(cmd)
    process = run(cmd, shell=True, capture_output=True)
    files = process.stdout.decode('utf-8').strip().split("\n")

    for i, file in enumerate(files):
        print("="*10+"\t{}/{}\t".format(i+1, len(files))+"="*10)
        print(file)
        repo = basename(dirname(file))
        with open(file, 'r') as f:
            lines = f.readlines()

        fns = []
        authors = []
        for line in lines:
            fn, author = tuple(line.split(": "))
            fns.append(fn)
            authors.append(author)

        authors = list(set(authors))
        with open(stat_info_file, 'a') as f:
            f.write("{} -- \tauthors:- {}\tfuns:- {}\taverage:- {}\n".format(repo, len(authors), len(fns), len(fns)/len(authors)))

def merge_num_fn_info(json_file):
    # if an author contributes multi-bin, then we need to merge them.
    with open(json_file, 'r') as f:
        author_fn_num_info = json.load(f)

    author_fn_num_dict = dict()
    for binary in author_fn_num_info.keys():
        for author in author_fn_num_info[binary]:
            if author in author_fn_num_dict.keys():
                print("{} in {} conflict!!".format(author, binary))
                # author_1 = author+"_1"
                author_fn_num_dict[author] += author_fn_num_info[binary][author]
            else:
                author_fn_num_dict[author] = author_fn_num_info[binary][author]

        # author_fn_num_dict = {**author_fn_num_info[binary], **author_fn_num_dict}

    sorted_author_fn_num_dict = dict(sorted(author_fn_num_dict.items(), key=lambda item: item[1]))
    # print(json.dumps(sorted_author_fn_num_dict, indent=4))
    return sorted_author_fn_num_dict

"""
get all folders of a directory, then get author names and fns
print the number of authors and funs
input : a path of a directory
return: None
"""
def get_fn_num_from_dir(directory):
    cmd = "find "+directory+" -mindepth 1 -maxdepth 1 -type d "
    process = run(cmd, shell=True, capture_output=True)
    subdirs = process.stdout.decode('utf-8').strip().split("\n")

    fns = []
    authors = []
    for subdir in subdirs:
        _, author, _, fn_name = get_fname_author_bin_from_parts(basename(subdir).split("_"))
        fns.append(fn_name)
        authors.append(author)

    print("In total: {} fn;\t{} authors".format(len(list(set(fns))), len(list(set(authors)))))
    # print(list(set(authors)))

def add_element_as_list_2_key_in_dict(element, key, dictionary):
    if key in dictionary.keys():
        dictionary[key] += [element]
    else:
        dictionary[key] = [element]
    # return dictionary

"""
read "db_info.json" file to get content of "all_fns"
get author names from each element
input: a path of "db_info.json"
return: a list of author names
"""
def get_authors_from_dbinfo(db_info_path):
    with open(db_info_path, 'r') as f:
        data = json.load(f)

    authors = []
    for s in data["all_fns"]:
        _, author, _, _ = get_names_from_normal_parts(s.split("_"))
        authors.append(author)

    authors = list(set(authors))
    return authors

"""
find all folders in a directory  
get author name and fn name from each foler
input: a path of a directory
return: a pair of a list of author names and dictionary about author and their fn number
"""
def get_authors_from_dir(dir_path):
    cmd = "find " + dir_path + " -mindepth 1 -maxdepth 1 -type d "
    find_proc = run(cmd, shell=True, capture_output=True)
    slices_list = find_proc.stdout.decode('utf-8').strip().split('\n')

    authors = []
    stat_dict = {}
    # print(cmd, '\n', slices_list)
    for s_path in slices_list:
        try:
            fn, author, _, _ = get_fname_author_bin_from_parts(basename(s_path).split("_"))
        except Exception as e:
            print(s_path)
            raise e
        authors.append(author)
        add_element_as_list_2_key_in_dict(fn, author, stat_dict)


    authors = list(set(authors))
    stat_dict = {author: len(list(set(stat_dict[author]))) for author in stat_dict.keys()}

    return authors, stat_dict