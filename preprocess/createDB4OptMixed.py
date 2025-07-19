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
        self.srcDirPath = join(dbBasePath, 'src')

        self.srcInfoJson = join(self.srcDirPath, 'src_Info.json')

        if not exists(self.binDirPath):
            run('mkdir -p '+self.binDirPath, shell=True)
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

    def getSrcCode(self):
        # srcFiles = [join(self.srcDirPath, f) for f in os.listdir(self.srcDirPath) if isfile(join(self.srcDirPath, f))]
        # return srcFiles
        with open(self.srcInfoJson, 'r') as f:
            authorsCodeDict = json.load(f)

        srcFiles = []
        for author in authorsCodeDict.keys():
            for src in authorsCodeDict[author]['src']:
                srcFiles.append(join(self.srcDirPath, author+"_"+src))

        return srcFiles

    def pickupBin(self):
        """
            choose authors and their binaries based on requirments
        """
        # choose authors who have enough files 
        candidates = {author:binaries for (author, binaries) in self.binInfo.items() if len(binaries)/4 >= self.arg.filesPerAuthor}

        # randomly choose authors
        if len(candidates.keys()) >= self.arg.numAuthors:
            authors = random.sample(list(candidates.keys()), self.arg.numAuthors)
            candidates = {author:candidates[author] for author in authors}
        else:
            print("only {} candidates meet requirment".format(len(candidates)))
            return
        
        if self.arg.exact:
            # the size of binaries exactly equal to the size of files per author 
            # take opt0 part first
            candidates = {author:binaries[:self.arg.filesPerAuthor] for (author, binaries) in candidates.items()}
            # expand to opt1 - opt3 parts
            for author in candidates.keys():
                binaries = candidates[author]
                newBinaries = []
                for binary in binaries:
                    # print(binary)
                    newBinaries.append(binary)
                    for i in range(1,4):
                        newBin = binary[:-1] + str(i)
                        print(newBin)
                        if newBin in self.binInfo[author]:
                            newBinaries.append(newBin)
                        else:
                            print("{} do not exist".format(newBin))
                            # candidates[author].remove(bin)
                candidates[author] = newBinaries
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

    def compileSrcFile(self, optLv, src):
        lv = optLv[-1]
        if src.lower().endswith('c'):
            compiler = 'gcc'
        else:
            compiler = 'g++'

        # binary_name includes authors' name, end with '-optX'
        binary_name = splitext(basename(src))[0]+'-opt'+lv
        bin_path = join(self.binDirPath, binary_name)
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
        
    def compilation(self, srcFiles):
        # compile all src code files that you get
        # record authors and their binaries
        if exists(self.binInfoJson):
            return
        if self.arg.sameas != None:
            pass
            # # read db info file
            # with open(self.srcdbInfoJson, 'r') as datasetInfo:
            #     data = json.load(datasetInfo)
            # p_i = list(data.keys())

        # compile src for diff opt levels. bin names end with '-optX'
        if not self.arg.mixed:
            optNum = 1
        else:
            optNum = len(self.optLvs) 
        for optLv in self.optLvs[:optNum]:
            # parallelly run compilation
            arguments = [(optLv, src) for src in srcFiles]
            pool = multiprocessing.Pool()
            compile_rslts = pool.starmap(self.compileSrcFile, arguments)
            pool.close()
            pool.join()

            # recording authors and corresponding binaries paths
            for e in compile_rslts:
                if e[1] == None:
                    continue
                else:
                    src, bin_path = e
                    file_name = splitext(basename(src))[0]
                    author = "_".join(file_name.split("_")[:-1])
                    if author not in self.binInfo.keys():
                        self.binInfo[author] = [bin_path]
                    else:
                        self.binInfo[author].append(bin_path)

        # save db info to a json file
        with open(self.binInfoJson, 'w') as binInfoJson:
            json.dump(self.binInfo, binInfoJson, indent=4)

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('-m', "--model", type=str, help="ndss18 or binauthor or eigen", required=True)
    parser.add_argument("--dbroot", type=str, help="path to your db root", required=True, default=None, dest='dbroot')
    parser.add_argument("--numAuthors", type=int, help="the number of authors in the dataset", default=1000)
    parser.add_argument("--target", type=str, help="target dir under db root", required=True, dest='target')
    parser.add_argument("--numFiles", type=int, help="the number of src files per author", default=10, dest='filesPerAuthor')
    parser.add_argument("--sameas", type=str, help="use the same dataset Json file", default=None, dest='sameas')
    parser.add_argument("--exact", help="if exactly choose # files per author, default=True", action='store_true', dest='exact')
    parser.add_argument("--no-exact", help="if exactly choose # files per author, default=True", action='store_false', dest='exact')
    parser.add_argument("--mixed", help="if mixed different opt-lv, default=False", action='store_true', dest='mixed')
    parser.set_defaults(exact=True, mixed=False)
    args = parser.parse_args()

    dbObj = createMixedDB(args)
    # x = input()
    # srcFiles = dbObj.getSrcCode()
    # print("getting source code done")
    # dbObj.compilation(srcFiles)
    # print("compilation finished")
    dbObj.pickupBin()
    print("choose authors and binaries done")
    dbObj.mkDB()
    print("all binaries have been moved to the target.")
