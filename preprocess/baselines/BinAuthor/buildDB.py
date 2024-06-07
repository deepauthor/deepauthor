import os, sys, argparse
from subprocess import run
import json, random, time
from os.path import join, exists, dirname, basename, isdir, isfile
from os import listdir


class db4BinAuthor():
    """docstring for db4BinAuthor"""
    def __init__(self, args):
        self.args = args
        self.target_dir = args.targetDir
        # self.db_from_file = args.dbFrom
        self.db_name = args.dbname
        self.authors_exe_info = join(args.path_gcjexe, 'bin_info.json')
        self.dataset_info = join(self.target_dir, self.db_name, self.db_name+"_datasetInfo.json")
        # self.last_iter = time.time()
        if not exists(join(self.target_dir, self.db_name)):
            os.makedirs(join(self.target_dir, self.db_name))
        # print("json file path :{}".format(self.authors_tasks_info))
        # print("target dir of exe files : {}".format(self.target_dir))

    def collection(self):
        """ move files in 'train' to 'targetDir/AuthorsDirectory/author/exe'
            move files in 'test' to 'targetDir/AuthorsDirectoryTest/author/exe'
        """
        print("collecting files >>> ")
        with open(self.dataset_info, 'r') as train_test:
            data = json.load(train_test)
        if 'train' in data.keys() or 'test' in data.keys():
            train_files = data['train']
            test_files = data['test']
            # on windows move command need target dir path but not full file path
            # make sure target dir exists
            for item in train_files:
                target_dir = join(self.target_dir, self.db_name, 'AuthorsDirectory', item['author'])
                if not exists(target_dir):
                    os.makedirs(target_dir)
                for file in item['files']:
                    # cmd = "move "+ file + " "+ target_dir
                    cmd = "copy "+ file + " "+ target_dir
                    run(cmd, shell=True)

            for item in test_files:
                target_dir = join(self.target_dir, self.db_name, 'AuthorsDirectoryTest', item['author'])
                if not exists(target_dir):
                    os.makedirs(target_dir)
                for file in item['files']:
                    # cmd = "move "+ file + " "+ target_dir
                    cmd = "copy "+ file + " "+ target_dir
                    run(cmd, shell=True)

    def buildDBInfoFromJson(self):
        destination = join(self.target_dir, self.db_name)
        if exists(self.dataset_info):
            return
        if not exists(destination):
            os.makedirs(destination)
            # return
        # author exe info json is the db info json file
        with open(self.authors_exe_info, 'r') as f:
            authors_exe_info = json.load(f)

        # record training and testing exe files
        dataset_dict = {'train':[], 'test':[]}
        for author in authors_exe_info.keys():
            unit = len(authors_exe_info[author]) // 10
            dataset_dict['train'].append({'author':author, 'files':authors_exe_info[author][:9*unit]}) 
            dataset_dict['test'].append({'author':author, 'files':authors_exe_info[author][9*unit:]})            
        # print(self.dataset_info)
        with open(self.dataset_info,'w+') as f:
            json.dump(dataset_dict, f, indent=4)
        print("datasetFile : {}".format(self.dataset_info))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument("--dbFrom", type=str, help="path to db info json file that is used to build db for binauthor", required=True)
    parser.add_argument("--exe", type=str, help="path to gcj-exe", dest='path_gcjexe', required=True)
    parser.add_argument('-n', "--dbname", type=str, help="the name of db", required=True)
    parser.add_argument('-t', "--targetDir", type=str, help="target dir for exe files", required=True, dest='targetDir')
    # parser.add_argument('-nf', "--numFiles", type=int, help="the number of files per author", default=10)
    args = parser.parse_args()

    binauthor = db4BinAuthor(args)
    binauthor.buildDBInfoFromJson()
    print("choosd files")
    binauthor.collection()