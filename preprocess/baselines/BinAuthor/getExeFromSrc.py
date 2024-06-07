from os.path import join, exists, basename, dirname, abspath, splitext, isfile
import os, sys, multiprocessing
from subprocess import run
import csv, argparse, json
sys.path.append((os.path.dirname(os.path.abspath("__file__"))))
from Util import SrcFiles

class SrcFile4BinAuthor(SrcFiles):
    """docstring for SrcFile"""
    def __init__(self, arg):
        super(SrcFile4BinAuthor, self).__init__(arg)
        self.arg = arg
        if arg.fromDB == None:
            self.srcFiles = [join(self.srcDirPath, src) for src in os.listdir(arg.srcDir) if isfile(join(arg.srcDir, src))]
        else:
            self.srcFiles = []
        # self.binDirPath = arg.binDir
        # if not exists(self.binDirPath):
        #     # run('mkdir -p '+self.binDirPath, shell=True)
        #     run('mkdir  '+self.binDirPath, shell=True)
        # self.binInfoJson = join(self.binDirPath, "bin_Info.json")
        # if not exists(self.binInfoJson):
        #     self.binInfo = dict()
        # else:
        #     with open(self.binInfoJson, 'r') as f:
        #         self.binInfo = json.load(f)
        # self.optLvs = ['-O0', '-O1', '-O2', '-O3']
        # self.ldOpts = " -lm -lc -lpthread -lstdc++ -ldl "
        self.srcInfoJson = join(self.srcDirPath, 'src_Info.json')

    # def compile(self):
    #     # compile all src code files that you get
    #     # record authors and their binaries into 'bin_info.json'

    #     # compile src for diff opt levels. bin names end with '-optX'
    #     if not self.arg.mixed:
    #         optNum = 1
    #     else:
    #         optNum = len(self.optLvs) 
    #     for optLv in self.optLvs[:optNum]:
    #         # parallelly run compilation
    #         print('now working for {}'.format(optLv))
    #         arguments = [(optLv, src) for src in self.srcFiles]
    #         pool = multiprocessing.Pool()
    #         compile_rslts = pool.starmap(self.compileSrcFile, arguments)
    #         pool.close()
    #         pool.join()

    #         # recording authors and corresponding binaries paths
    #         for e in compile_rslts:
    #             if e[1] == None:
    #                 # compilation failed
    #                 continue
    #             else:
    #                 src, bin_path = e
    #                 file_name = splitext(basename(src))[0]
    #                 author = "_".join(file_name.split("_")[:-1])
    #                 if author not in self.binInfo.keys():
    #                     self.binInfo[author] = [bin_path]
    #                 else:
    #                     # self.binInfo may read from a JSON file
    #                     if bin_path not in self.binInfo[author]:
    #                         self.binInfo[author].append(bin_path)

    #     # save all authors and corresponding binaries info to a json file
    #     with open(self.binInfoJson, 'w') as binInfoJson:
    #         json.dump(self.binInfo, binInfoJson, indent=4)
    #     print("saved all authors and corresponding binaries into {}".format(self.binInfoJson))

    def getSrcPathFromDB(self):
        with open(self.arg.fromDB, 'r') as f:
            dbInfo = json.load(f)
        with open(self.srcInfoJson, 'r') as src:
            authorSrcDict = json.load(src)

        for author in dbInfo.keys():
            tmp = set([basename(binPath)[:-len('-opt0')] for binPath in dbInfo[author]])
            # print(tmp)
            for e in list(tmp):
                srcName = e.split('_')[-1]
                author = "_".join(e.split('_')[:-1])
                # print(author, srcName)
                for idx, src in enumerate(authorSrcDict[author]['src']):
                    if srcName in src:
                        srcName = src
                        break
                binName = "_".join([author, srcName])
                # print(binName)
                self.srcFiles.append(join(self.srcDirPath, binName))

    def compileSrcFile(self, optLv, src):
        lv = optLv[-1]
        if src.lower().endswith('c'):
            compiler = 'gcc'
        else:
            compiler = 'g++'

        # binary_name includes authors' name, end with '-optX'
        binary_name = splitext(basename(src))[0]+'-opt'+lv+'.exe'
        bin_path = join(self.binDirPath, binary_name)
        if exists(bin_path):
            return src, bin_path
        # cmd = compiler + " -m32 " + optLv + " " + src + " -o " + bin_path + self.ldOpts
        cmd = compiler + " -m32 " + optLv + " " + src + " -o " + bin_path 
        p = run(cmd, capture_output=True, shell=True)
        if p.returncode != 0:
            with open("compile_err1.log", 'a') as f:
                f.write("working for {}\ncmd : {}\n".format(src, cmd))
                f.write(p.stderr.decode('utf-8')+'\n')
            return src, None
        else:
            # record new binary path to the db json file
            return src, bin_path

class SrcFile4BinAuthor_2017(SrcFile4BinAuthor):
    """docstring for SrcFile4BinAuthor_2017"""
    def __init__(self, arg):
        super(SrcFile4BinAuthor_2017, self).__init__(arg)
        self.arg = arg
        if not exists(self.srcInfoJson):
            self.getSrcInfo()
        
    def getSrcInfo(self):
        folders = [join(self.srcDirPath, src) for src in os.listdir(self.arg.srcDir) if not isfile(join(self.arg.srcDir, src))]
        print("len foders: {}".format(len(folders)))
        for folder in folders:
            forlder = join(self.srcDirPath, folder)
            self.srcFiles += [join(folder, src) for src in os.listdir(folder) if isfile(join(folder, src))]
        print("len srcFiles :{} ".format(len(self.srcFiles)))
        authorSrcDict = dict()

        for src in self.srcFiles:
            author = splitext(src)[0].split("_")[-1]
            if author not in authorSrcDict.keys():
                authorSrcDict[author] = [src]
            else:
                authorSrcDict[author].append(src)

        with open(self.srcInfoJson, 'w') as f:
            json.dump(authorSrcDict, f, indent=4)

    def getSrcPathFromDB(self):
        self.srcFiles = []
        with open(self.arg.fromDB, 'r') as f:
            dbInfo = json.load(f)
        with open(self.srcInfoJson, 'r') as src:
            authorSrcDict = json.load(src)

        for author in dbInfo.keys():
            self.srcFiles += authorSrcDict[author]

    def getAuthor(self, src):
        file_name = splitext(basename(src))[0]
        author = (file_name.split("_")[-1])
        return author

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, help="path to src folder", dest='srcDir')
    parser.add_argument("--exe", type=str, help="path to exe folder", dest='binDir')
    parser.add_argument("--db", type=str, help="path to a db info json file that will be used for binAuthor", dest='fromDB')
    parser.add_argument("--mixed", help="if mixed different opt-lv, default=False", action='store_true', dest='mixed')
    parser.set_defaults(mixed=False,
                        fromDB=None,
                        srcDir='Y:\\gcj-src',
                        exeDir='Y:\\gcj-exe')
    args = parser.parse_args()
    # srcObj = SrcFile(args)
    srcObj = SrcFile4BinAuthor_2017(args)
    if args.fromDB != None:
        srcObj.getSrcPathFromDB()
    srcObj.compile()
