import os, sys, argparse, random, multiprocessing
from subprocess import run
import json, random, time
from os.path import join, exists, dirname, basename, isdir, isfile, splitext
from os import listdir
from sys import platform

class createMixedDB():
    """docstring for createDB
       choosing source files and then compile with different optimization level
       from -O0 to -O3. That means the size of dataset will be four times of the 
       product of authors and files per author.
    """
    def __init__(self, args):
        self.arg = args
        dbBasePath = args.dbroot
        self.target = join(dbBasePath, args.target)
        if not exists(self.target):
            run("mkdir -p "+self.target, shell=True)
        self.dbInfoJson = join(self.target, args.target+"_dbInfo.json")
        if exists(self.dbInfoJson):
            with open(self.dbInfoJson, 'r') as f:
                self.dbInfo = json.load(f)
        else:
            self.dbInfo = None
        # self.srcDirPath = '/media/yihua/2TB/backup/gcj-src/'
        # self.srcInfoJson = join(self.srcDirPath, 'src_Info.json')
        # self.binDirPath = '/media/yihua/2TB/backup/gcj-bin/'
        self.binDirPath = args.binDirPath
        
        # self.srcDirPath = join(dbBasePath, 'src')
        # if not exists(self.binDirPath):
        #     run('mkdir -p '+self.binDirPath, shell=True)
        assert exists(self.binDirPath), 'there must be a folder containing all binaries.'
        self.optLvs = ['-O0', '-O1', '-O2', '-O3']
        self.binInfoJson = join(self.binDirPath, "bin_Info.json")
        # if not exists(self.binInfoJson):
        #     self.binInfo = dict()
        # else:
        assert exists(self.binInfoJson), 'there must exist a JSON file containing binary info'
        with open(self.binInfoJson, 'r') as f:
            self.binInfo = json.load(f)

        self.ldOpts = " -lm -lc -lpthread -lstdc++ -ldl "

    def pickupBin(self):
        """
            choose authors and their binaries based on requirments
        """
        # choose authors who have enough files, which is based on mixed or not
        # candidates = {author:binaries for (author, binaries) in self.binInfo.items() if len(binaries)/4 >= self.arg.filesPerAuthor}
        if self.arg.mixed:
            optLv = '-opt3'
        else:
            optLv = '-opt0'
        candidates = {author:binaries for (author, binaries) in self.binInfo.items() if len([b for b in binaries if basename(b).endswith(optLv)]) > self.arg.filesPerAuthor}

        # randomly choose authors
        if len(candidates.keys()) >= self.arg.numAuthors:
            authors = random.sample(list(candidates.keys()), self.arg.numAuthors)
            candidates = {author:candidates[author] for author in authors}
        else:
            print("only {} candidates meet requirment".format(len(candidates)))
            return
        
        if self.arg.exact: 
            if self.arg.mixed:
                # the size of binaries exactly equal to the size of files per author 
                # take opt3 part first
                # candidates = {author:binaries[:self.arg.filesPerAuthor] for (author, binaries) in candidates.items()}
                candidates = {author: [b for b in binaries if basename(b).endswith(optLv)][:self.arg.filesPerAuthor] for (author, binaries) in candidates.items()}
                # expand to opt1 - opt3 parts
                for author in candidates.keys():
                    binaries = candidates[author]
                    newBinaries = []
                    for binary in binaries:
                        # print(binary)
                        newBinaries.append(binary)
                        for i in range(0,3):
                            newBin = binary[:-1] + str(i)
                            print(newBin)
                            if newBin in self.binInfo[author]:
                                newBinaries.append(newBin)
                            else:
                                print("{} do not exist".format(newBin))
                                # candidates[author].remove(bin)
                    candidates[author] = newBinaries
            else:
                # only take opt0 part
                candidates = {author: [b for b in binaries if basename(b).endswith(optLv)][:self.arg.filesPerAuthor] for (author, binaries) in candidates.items()}

        # the minimal size of the binaries meet the size of files per author
        self.dbInfo = candidates

    def move2DB(self):
        # for ndss 18
        # 'x' cross-fold validation -- 'px':[bin files]
        # db at ./gcj2020data/ndss18/target_dir
        # structure : target_dir/AuthorsDirectory/A_i/bin
        print("collecting files >>> ")
        # with open(self.destdbInfoJson, 'r') as train_test:
        #     data = json.load(train_test)
        data = self.dbJson
        with open(join(self.dbBasePath, "codeWithAuthDict.json"), 'r') as f:
            codeWithAuthDict = json.load(f)
        base_dir = join(self.dbBasePath, 'gcj2020data', 'ndss18')
        target_dir = join(base_dir, self.arg.targetDir, 'AuthorsDirectory')
        if not exists(target_dir):
            os.makedirs(target_dir)
        for p in data.keys():
            # check if target dir exists
            p_dir = join(target_dir, p)
            if not exists(p_dir):
                os.makedirs(p_dir)
                print("make a new dir: {}".format(p_dir))

            for file in data[p]:
                author = codeWithAuthDict[basename(file)]
                author_dir = join(p_dir, author)
                if not exists(author_dir):
                    os.makedirs(author_dir)
                if ',' in basename(file):
                    dest = join(author_dir, basename(file).replace(',','.'))
                else:
                    dest = join(author_dir, basename(file))
                cmd = 'cp '+file+' '+dest
                process = run(cmd, shell=True)
                if process.returncode == 0:
                    print("{} is copied to {}".format(file, dest))
                else:
                    print("copy failed")

    def mkDB(self):
        assert self.dbInfo is not None, "need to pickup binaries first"
        # mv binaries to target dir
        for author in self.dbInfo.keys():
            # create a folder for each author
            author_dir = join(self.target, 'AuthorsDirectory', author)
            if not exists(author_dir):
                run("mkdir -p "+author_dir, shell=True)

            # cp all binaries to the author's folder
            dest = author_dir
            for bin_path in self.dbInfo[author]:
                cmd = 'cp '+bin_path+' '+dest
                process = run(cmd, shell=True)
                if process.returncode == 0:
                    print("{} is copied to {}".format(bin_path, dest))
                else:
                    print("{} copy failed".format(bin_path))

        # save db info as a json file
        with open(self.dbInfoJson, 'w') as dbInfoJson:
            json.dump(self.dbInfo, dbInfoJson, indent=4)

class BuildDB4Binauthor_2017(createMixedDB):
    """docstring for BuildDB4Binauthor_2017"""
    def __init__(self, arg):
        super(BuildDB4Binauthor_2017, self).__init__(arg)
        self.arg = arg
        self.dbInfo = dict()
       
    def pickupBin(self):
        with open(self.arg.sameas, 'r') as f:
            db = json.load(f)

        with open(self.binInfoJson, 'r') as f:
            binInfo = json.load(f)

        for author in db.keys():
            if author not in self.dbInfo.keys():
                self.dbInfo[author] = binInfo[author]

    def mkDB(self):
        assert self.dbInfo is not None, "need to pickup binaries first"
        # mv binaries to target dir
        for author in self.dbInfo.keys():
            # create a folder for each author
            author_dir = join(self.target, 'AuthorsDirectory', author)
            if not exists(author_dir):
                run("mkdir -p "+author_dir, shell=True)
            author_dir_test = join(self.target, 'AuthorsDirectoryTest', author)
            if not exists(author_dir_test):
                run("mkdir -p "+author_dir_test, shell=True)

            # cp all binaries to the author's folder
            dest = author_dir
            for bin_path in self.dbInfo[author][:-1]:
                cmd = 'copy '+bin_path+' '+join(dest, basename(bin_path))
                process = run(cmd, shell=True)
                if process.returncode == 0:
                    print("{} is copied to {}".format(bin_path, dest))
                else:
                    print("{} copy failed".format(bin_path))
            dest = author_dir_test
            cmd = 'copy '+self.dbInfo[author][-1]+' '+join(dest, basename(self.dbInfo[author][-1]))
            process = run(cmd, shell=True)
            if process.returncode == 0:
                print("{} is copied to {}".format(bin_path, dest))
            else:
                print("{} copy failed".format(bin_path))

        # save db info as a json file
        with open(self.dbInfoJson, 'w') as dbInfoJson:
            json.dump(self.dbInfo, dbInfoJson, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('-m', "--model", type=str, help="ndss18 or binauthor or eigen", required=True)
    parser.add_argument("--dbroot", type=str, help="path to your db root", required=True, default=None, dest='dbroot')
    parser.add_argument("--numAuthors", type=int, help="the number of authors in the dataset", default=1000)
    parser.add_argument("--dbname", type=str, help="target dir under db root", required=True, dest='target')
    parser.add_argument("--numFiles", type=int, help="the number of src files per author", default=10, dest='filesPerAuthor')
    parser.add_argument("--sameas", type=str, help="use the same dataset Json file", default=None, dest='sameas')
    parser.add_argument("--exact", help="if exactly choose # files per author, default=True", action='store_true', dest='exact')
    parser.add_argument("--no-exact", help="if exactly choose # files per author, default=True", action='store_false', dest='exact')
    parser.add_argument("--mixed", help="if mixed different opt-lv, default=False", action='store_true', dest='mixed')
    parser.add_argument("--exe", "--bin", type=str, help="path to all exe files", dest="binDirPath")
    parser.set_defaults(exact=True, mixed=False)
    args = parser.parse_args()

    # dbObj = createMixedDB(args)
    dbObj = BuildDB4Binauthor_2017(args)
    # x = input()
    # srcFiles = dbObj.getSrcCode()
    # print("getting source code done")
    # dbObj.compilation(srcFiles)
    # print("compilation finished")
    dbObj.pickupBin()
    print("choose authors and binaries done")
    dbObj.mkDB()
    print("all binaries have been moved to the target.")
