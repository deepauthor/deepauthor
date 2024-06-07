from Util import SrcFiles, BinFiles
from os.path import join, basename, exists
import argparse, json, random
from subprocess import run

class MyBinFiles(BinFiles):
    """docstring for MyBinFiles"""
    def __init__(self, args):
        super(MyBinFiles, self).__init__(args)
        self.args = args

    def buildDB(self):
        self.createDBInfoJson()
        self.mvBin()
        
    def createDBInfoJson(self):
        db_dir = join(self.dbRoot, self.args.dbName)
        if not exists(db_dir):
            run("mkdir -p "+db_dir, shell=True)
        dbInfo_file = join(db_dir, "_".join([self.args.dbName, "dbInfo.json"]))
        with open(self.args.dbInfo, 'r') as f:
            data = json.load(f)

        dbInfo = dict()
        for author in data.keys():
            dbInfo[author] = random.sample(data[author], self.args.n_files)

        with open(dbInfo_file, 'w') as f:
            json.dump(dbInfo, f, indent=4)
        self.dbInfo = dbInfo
        
    def mvBin(self):
        for author in self.dbInfo.keys():
            dest_dir = join(self.dbRoot, self.args.dbName, "AuthorsDirectory", author)
            if not exists(dest_dir):
                run("mkdir -p "+dest_dir, shell=True)

            for binary in self.dbInfo[author]:
                dest = join(dest_dir, basename(binary))
                if exists(dest):
                    continue
                else:
                    cmd = "cp "+binary+" "+dest
                    print(cmd)
                    run(cmd, shell=True)

def buildDB(args):
    bin_files = MyBinFiles(args)
    bin_files.buildDB()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for building DB task from a existed db.')
    parser.add_argument("--dbInfo", type=str, help="path to an existed db info json file", )
    parser.add_argument("--dbDir", type=str, help="dir of db root", ) 
    parser.add_argument("--binDir", type=str, help="dir of all binaries", ) 
    parser.add_argument("--dbName", type=str, help="name of the db", )
    parser.add_argument("--n_files", type=int, help="num of binaries", )
    args = parser.parse_args()
    buildDB(args)