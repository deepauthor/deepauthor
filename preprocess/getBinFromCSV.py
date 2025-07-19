from os.path import join, exists, basename, dirname, abspath, splitext, abspath
from subprocess import run
import csv, argparse
import json
import os, sys, multiprocessing
from os.path import join
from Util import CSVFile, SrcFile, BinFile


def getNcompileSrcCodeByNum():
    parser = argparse.ArgumentParser()
    parser.add_argument("--srcDir", type=str, help="source code output dir", default='/media/2TB/backup/gcj-src/')
    parser.add_argument("--binDir", type=str, help="bin output dir", default='/media/2TB/backup/gcj-bin/')
    parser.add_argument("--csvDir", type=str, help="point to all csv files dir", default=None)
    parser.add_argument("--csvFile", type=str, help="point to a csv file", default=None)
    parser.add_argument('--fileNum', type=int, help="the min num of files for each author.", required=True)
    parser.add_argument('--authorNum', type=int, help="the number of authors.", required=True)
    parser.add_argument('--years', type=int, help="the number of years you want to be over.", default=1)
    parser.add_argument("--mixed", help="if mixed different opt-lv, default=False", action='store_true', dest='mixed')
    args = parser.parse_args()

    cmd = 'find '+args.csvDir +" -type f -iname '*.csv'"
    process = run(cmd, shell=True, capture_output = True)
    allCSVFiles = process.stdout.decode('utf-8').split('\n')[:-1]
    allCSVFiles.sort(reverse=True)
    srcInfoFile = join(args.srcDir, 'src_Info.json')
    if not exists(srcInfoFile):
        authorsWithCodeDict = dict()
    else:
        with open(srcInfoFile, 'r') as f :
            authorsWithCodeDict = json.load(f)

    overyears = args.years
    yourChoice = allCSVFiles[:overyears]
    print("your choice: {}".format(yourChoice))
    # for csvfile in allCSVFiles:
    for csvfile in yourChoice:
        print("working on {}".format(csvfile))
        csvFile = CSVFile(args, csvfile)
        tempDict = csvFile.getAllCodeByNum(args.fileNum)
        # authorsWithCodeDict.update(tempDict)
        # update 
        for author in tempDict:
            if author in authorsWithCodeDict.keys():
                src = list(set(authorsWithCodeDict[author][csvFile.str_content] + tempDict[author][csvFile.str_content]))
                authorsWithCodeDict[author] = {csvFile.str_num:len(src), csvFile.str_content:src}
            else:
                authorsWithCodeDict[author] = tempDict[author]
        if len(authorsWithCodeDict.keys()) > args.authorNum:
            break

    if len(authorsWithCodeDict.keys()) < args.authorNum:
        print("Failed: Only {} authors meeting requirements".format(len(authorsWithCodeDict.keys())))
        return

    print("finally, there are {} authors".format(len(authorsWithCodeDict.keys())))
    with open(srcInfoFile, 'w') as f:
        json.dump(authorsWithCodeDict, f, indent=4)
    print("src info saved into {}".format(join(args.srcDir, 'src_Info.json')))

    print("compiling source code files to binaries")
    allSrcFiles = []
    for author in authorsWithCodeDict.keys():
        allSrcFiles += [join(abspath(args.srcDir), author+"_"+src) for src in authorsWithCodeDict[author][csvFile.str_content]]
    srcFiles = SrcFile(args, allSrcFiles)
    srcFiles.compile()
    print("finished")

def getInfo4CCode():
    parser = argparse.ArgumentParser()
    parser.add_argument("--srcDir", type=str, help="source code output dir", default=None)
    parser.add_argument("--csvDir", type=str, help="point to all csv files dir", default=None)
    # parser.add_argument("--csvFile", type=str, help="point to a csv file", default=None)
    # parser.add_argument('--fileNum', type=int, help="the min num of files for each author.", required=True)
    # parser.add_argument('--authorNum', type=int, help="the number of authors.", required=True)
    # parser.add_argument('--overyears', type=int, help="the number of years you want to be over.", default=0)
    parser.add_argument('--json', type=str, help="point to a json file to save the output", default="data.json")
    args = parser.parse_args()

    cmd = 'find '+args.csvDir +" -type f -iname '*.csv'"
    process = run(cmd, shell=True, capture_output = True)
    allCSVFiles = process.stdout.decode('utf-8').split('\n')[:-1]
    allCSVFiles.sort(reverse=True)
    print(allCSVFiles)
    infoByYears = dict()
    for csvfile in allCSVFiles:
        print("working on {}".format(csvfile))  
        csvFile = CSVFile(args, csvfile)
        tempDict = csvFile.getStatistic4C()
        infoByYears.update(tempDict)
    
    data = dict()
    for year in infoByYears.keys():
        for author in infoByYears[year].keys():
            if author in data.keys():
                data[author] += infoByYears[year][author]
            else:
                data[author] = infoByYears[year][author]

    if not exists(args.json):
        run("mkdir -p "+args.json, shell=True)
        
    with open(args.json, "w") as f:
        json.dump(data, f, indent=4)
    num = [len(data[author]) for author in data.keys()]
    num.sort()

    print("finally, there are {} authors, and {} files in total".format(len(data.keys()), sum(num)))
    print("top 5 {}".format(num[-5:]))

if __name__ == '__main__':
    # getInfo4CCode()
    getNcompileSrcCodeByNum()
