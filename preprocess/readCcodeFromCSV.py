from os.path import join, exists, basename, dirname, abspath, splitext
from subprocess import run
import csv, argparse
import json
import os, sys
from os.path import join
from Util import CSVFile

def extractObjSrcCodeByNum():
    parser = argparse.ArgumentParser()
    parser.add_argument('-out', "--outCodeDir", type=str, help="source code output dir", required=True)
    parser.add_argument("--csvDir", type=str, help="point to all csv files dir", default=None)
    parser.add_argument("--csvFile", type=str, help="point to a csv file", default=None)
    parser.add_argument('--fileNum', type=int, help="the min num of files for each author.", required=True)
    parser.add_argument('--authorNum', type=int, help="the number of authors.", required=True)
    parser.add_argument('--overyears', type=int, help="the number of years you want to be over.", default=0)
    args = parser.parse_args()

    if args.csvDir != None:
        cmd = 'find '+args.csvDir +" -type f -iname '*.csv'"
        process = run(cmd, shell=True, capture_output = True)
        allCSVFiles = process.stdout.decode('utf-8').split('\n')[:-1]
        allCSVFiles.sort(reverse=True)
        print(allCSVFiles)

    if args.csvFile != None:
        allCSVFiles = [args.csvFile]
    authorsWithCodeDict = dict()
    overyears = args.overyears
    for csvfile in allCSVFiles:
        print("working on {}".format(csvfile))
        csvFile = CSVFile(args, csvfile)
        tempDict = csvFile.getAllCodeByNum(args.fileNum)
        authorsWithCodeDict.update(tempDict)
        if len(authorsWithCodeDict.keys())>args.authorNum:
            if len(tempDict.keys())==0:
                continue
            if overyears == 0:
                break
            else:
                overyears -= 1

    print("finally, there are {} authors".format(len(authorsWithCodeDict.keys())))
    with open(join(args.outCodeDir, 'src_Info.json'), 'w') as f:
        json.dump(authorsWithCodeDict, f, indent=4)
    print("src info saved")

def getInfo4CCode():
    parser = argparse.ArgumentParser()
    parser.add_argument('-out', "--outCodeDir", type=str, help="source code output dir", default=None)
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

def extractObjSrcCodeByAuthors():
    parser = argparse.ArgumentParser()
    parser.add_argument('-out', "--outCodeDir", type=str, help="source code output dir", required=True)
    parser.add_argument("--csvDir", type=str, help="point to all csv files dir", default=None)
    parser.add_argument("--csvFile", type=str, help="point to a csv file", default=None)
    parser.add_argument('--fileNum', type=int, help="the min num of files for each author.", required=True)
    parser.add_argument('--authors', type=str, help="path to a file containing authors.", required=True)
    parser.add_argument('--overyears', type=int, help="the number of years you want to be over.", default=0)
    args = parser.parse_args()


    with open(args.authors, 'r') as f:
        data = readlines()

    authors = data.strip().split('\n')

    if args.csvDir != None:
        cmd = 'find '+args.csvDir +" -type f -iname '*.csv'"
        process = run(cmd, shell=True, capture_output = True)
        allCSVFiles = process.stdout.decode('utf-8').split('\n')[:-1]
        allCSVFiles.sort(reverse=True)
        print(allCSVFiles)

    if args.csvFile != None:
        allCSVFiles = [args.csvFile]
    authorsWithCodeDict = dict()
    overyears = args.overyears
    for csvfile in allCSVFiles:
        print("working on {}".format(csvfile))
        csvFile = CSVFile(args, csvfile)
        tempDict = csvFile.getAllCodeByAuthors(args.authors)
        authorsWithCodeDict.update(tempDict)
        if len(authorsWithCodeDict.keys())>args.authorNum:
            if len(tempDict.keys())==0:
                continue
            if overyears == 0:
                break
            else:
                overyears -= 1

    print("finally, there are {} authors".format(len(authorsWithCodeDict.keys())))
    with open(join(args.outCodeDir, 'src_Info.json'), 'w') as f:
        json.dump(authorsWithCodeDict, f, indent=4)
    print("src info saved")
if __name__ == '__main__':
    # getInfo4CCode()
    extractObjSrcCodeByAuthors()