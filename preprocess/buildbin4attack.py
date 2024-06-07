from Util import SrcFiles, BinFiles

from os.path import join, exists, basename, dirname, abspath, splitext, abspath
from subprocess import run
import csv, argparse
import json, random
import os, sys, multiprocessing

class BinFiles4attack(BinFiles):
    """docstring for BinFiles4attack"""
    def __init__(self, arg):
        super(BinFiles4attack, self).__init__(arg)
        self.arg = arg
        self.dbInfoJson = join(self.dbRoot, "db_info.json")
        
    def buildDB(self):
        # fixedAuthors = ["ACMonster", "4yn", "chocimir", "ALOHA.Brcps"]
        db_info = dict() 
        # db_authors = set(fixedAuthors) | set(random.sample(set([e for e in self.binInfo.keys() if e not in fixedAuthors]), self.arg.num_authors-len(fixedAuthors)))
        db_authors = ['4yn', 'ACMonster', 'ALOHA.Brcps', 'Alireza.bh', 'DAle', 'ShayanH', 'SummerDAway', 'TungNP', 'aman.chandna', 'ccsnoopy', 'chocimir', 'csegura', 'eugenus', 'fragusbot', 'iPeter', 'jiian', 'liymouse', 'sdya', 'thatprogrammer', 'vudduu']
        for author in db_authors:
            author_root = join(self.dbRoot, author)
            if not exists(author_root):
                run("mkdir -p "+author_root, shell=True)
            binaries = self.binInfo[author]

            # move binaries
            db_info[author] = []
            for binary in binaries:
                bin_name = basename(binary)
                if self.arg.mixed == False and self.arg.opt != bin_name[-len("optx"):]:
                    continue
                dest = join(author_root, bin_name)
                cmd_mv = 'cp '+binary+' '+dest
                run(cmd_mv, shell=True)
                db_info[author].append(binary)

        with open(self.dbInfoJson, 'w') as f:
            json.dump(db_info, f, indent=4)

class SrcFiles4attack(SrcFiles):
    """docstring for SrcFiles4attack"""
    def __init__(self, arg):
        super(SrcFiles4attack, self).__init__(arg)
        self.arg = arg

    # def compile(self):
    #     # compile src for diff opt levels. bin names end with '-optX'
    #     if not self.arg.mixed:
    #         optNum = 1
    #     else:
    #         optNum = len(self.optLvs) 

    #     if exists(self.binInfoJson):
    #         print("{} exist, skip compilation step".format(self.binInfoJson))
    #         return

    #     for optLv in self.optLvs[:optNum]:
    #         # parallelly run compilation
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
    #                 # file_name: round_id + challeng_id + author_name
    #                 src, bin_path = e
    #                 # file_name = splitext(basename(src))[0]
    #                 # author = "_".join(file_name.split("_")[2:])
    #                 author = getAuthor(src)
    #                 if author not in self.binInfo.keys():
    #                     self.binInfo[author] = [bin_path]
    #                 else:
    #                     # self.binInfo may read from a JSON file
    #                     if bin_path not in self.binInfo[author]:
    #                         self.binInfo[author].append(bin_path)

    #     # save all authors and corresponding binaries info to a json file
    #     # read the json file first if exists
    #     if exists(self.binInfoJson):
    #         with open(self.binInfoJson, 'r') as binInfoJson:
    #             oldBinInfo = json.load(binInfoJson)
    #     else:
    #         oldBinInfo = dict()
    #     # combine binInfoJson with old one 
    #     for author in oldBinInfo.keys():
    #         if author not in binInfoJson.keys():
    #             binInfoJson[author] = oldBinInfo[author]
    #         else:
    #             binInfoJson[author] += oldBinInfo[author]
    #     # save the latest info
    #     with open(self.binInfoJson, 'w') as binInfoJson:
    #         json.dump(self.binInfo, binInfoJson, indent=4)
    #     print("saved all authors and corresponding binaries into {}".format(self.binInfoJson))

    def getAuthor(self, src):
        file_name = splitext(basename(src))[0]
        author = "_".join(file_name.split("_")[2:])
        return author
        

def buildbin():
    parser = argparse.ArgumentParser()
    parser.add_argument("--srcDir", type=str, help="source code dir path", default=None)
    parser.add_argument("--binDir", type=str, help="a path of binary files dir", default=None)
    parser.add_argument("--dbDir", type=str, help="point to db root", default=None, dest="dbDir")
    # parser.add_argument('--fileNum', type=int, help="the min num of files for each author.", required=True)
    parser.add_argument('--authorNum', type=int, help="the number of authors.", dest="num_authors", default=50)
    parser.add_argument("--mixed", help="if mixed different opt-lv, default=False", action='store_true', dest='mixed')
    args = parser.parse_args()

    allSrcFiles = SrcFiles4attack(args)
    allSrcFiles.compile()

    allBinFiles = BinFiles4attack(args)
    allBinFiles.buildDB()

class SrcFiles4attacked(SrcFiles):
    """docstring for SrcFiles4attacked"""
    def __init__(self, arg):
        super(SrcFiles4attacked, self).__init__(arg)
        self.arg = arg

    def getAuthor(self, src):
        file_name = splitext(basename(src))[0]
        author = (file_name.split("_")[-1])
        return author


class BinFiles4attacked(BinFiles):
    """docstring for BinFiles4attack"""
    def __init__(self, arg):
        super(BinFiles4attacked, self).__init__(arg)
        self.arg = arg
        self.dbInfoJson = join(self.dbRoot, "db_Info.json")
        
    def buildDB(self):
        db_info = dict() 
        db_authors = self.binInfo.keys()
        print(self.binInfo)
        for author in db_authors:
            dest_author_root = join(self.dbRoot, "AuthorsDirectory", author)
            if not exists(dest_author_root):
                run("mkdir -p "+dest_author_root, shell=True)
            binaries = self.binInfo[author]

            # move binaries
            db_info[author] = binaries
            for binary in binaries:
                bin_name = basename(binary)
                if self.arg.mixed == False and self.arg.opt != bin_name[-1]:
                    continue
                # opt = bin_name[-len("-optx"):]
                new_bin_name = bin_name
                dest = join(dest_author_root, new_bin_name)
                cmd_mv = 'cp '+binary+' '+dest
                run(cmd_mv, shell=True)
                # db_info[author].append(binary)

        self.updateJsonFile(self.dbInfoJson, db_info)
        # with open(self.dbInfoJson, 'w') as f:
        #     json.dump(db_info, f, indent=4)

def buildbin4Attacked():
    parser = argparse.ArgumentParser()
    parser.add_argument("--srcDir", type=str, help="source code dir path", default=None)
    parser.add_argument("--binDir", type=str, help="a path of binary files dir", default=None)
    parser.add_argument("--dbDir", type=str, help="point to db root", default=None, dest="dbDir")
    parser.add_argument("--mixed", help="if mixed different opt-lv, default=False", action='store_true', dest='mixed')
    parser.add_argument("--opt", type=str, help="opt lv", default="0", dest="opt")
    args = parser.parse_args()

    allSrcFiles = SrcFiles4attacked(args)
    allSrcFiles.compile()

    allBinFiles = BinFiles4attacked(args)
    allBinFiles.buildDB()

if __name__ == '__main__':
    # buildbin()
    buildbin4Attacked()