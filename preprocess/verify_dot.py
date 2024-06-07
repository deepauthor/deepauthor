import os, argparse, json, random
import networkx as nx
import numpy as np
from subprocess import run
import torch
# from unixcoder import UniXcoder
from os.path import exists, join, basename, splitext
import faulthandler

faulthandler.enable()

def createDataset(args):
    dotDirPath = os.path.abspath(args.dotpath)
    pdgFiles = ['0-pdg.dot','1-pdg.dot']
    astFiles = ['0-ast.dot','1-ast.dot']

    # get all pesudo sub dirs
    cmd = 'find ' + dotDirPath + ' -maxdepth 1 -type d'
    print(cmd)
    proc = run(cmd, capture_output=True, shell=True)
    # 1st element is the pdgDir; the last one is '\n'
    subDirs = proc.stdout.decode('utf-8').split('\n')[1:-1]
    subDirs = sorted(subDirs)
    print("there are {} subdirs".format(len(subDirs)))
    # print(subDirs)

    # get all authors for all slice files
    authors = ["_".join(basename(e).split("_")[2:-1]) for e in subDirs]
    tmp = set(authors)
    authors = sorted([ e for e in list(tmp) if e !=''])
    # print(authors)

    # 5 files will be recorded, open them first as append mode
    dataRoot = join(args.datadir, args.dataname)
    astDataRoot = join(dataRoot, args.dataname+'_ast')
    pdgDataRoot = join(dataRoot, args.dataname+'_pdg')
    if not exists(dataRoot):
        os.makedirs(dataRoot)
        os.makedirs(astDataRoot)
        os.makedirs(pdgDataRoot)
    # prefix = join(args.datadir, args.dataname, args.dataname)
    astPrefix = join(astDataRoot,basename(astDataRoot))
    pdgPrefix = join(astDataRoot,basename(astDataRoot))
    cmd = 'find ' + dataRoot + ' -type f'
    proc = run(cmd.split(' '), capture_output=True)
    files = proc.stdout.decode('utf-8').split('\n')[:-1]
    print("exists {} data files".format(len(files)))
    if len(files) > 0:
        run(['rm']+files)
        print("deleted existed files. Done.")

    relationalOp = ['greaterThan', 'greaterEqualsThan', 'equals', 'notEquals', 'lessThan', 'lessEqualsThan']

    # check API info for

    graphIndicator = 1
    nodeID_ast = 1
    nodeMax = 0
    nodeID_pdg = 1
    for cnt, sub in enumerate(subDirs):
        astPath = join(sub, 'ast')
        pdgPath = join(sub, 'pdg')
        astFile = join(astPath, astFiles[1])
        pdgFile = join(pdgPath, pdgFiles[1])
        if not exists(astFile):
            astFile = join(astPath, astFiles[0])
        if not exists(pdgFile):
            pdgFile = join(pdgPath, pdgFiles[0])
        print("working : {}".format(sub))
        # print("ast file {}".format(astFile))
        # print("pdg file {}".format(pdgFile))
        g_ast = nx.drawing.nx_agraph.read_dot(astFile)
        g_pdg = nx.drawing.nx_agraph.read_dot(pdgFile)

def chkDotCorret(args):
    dotDirPath = os.path.abspath(args.dotpath)
    pdgFiles = ['0-pdg.dot','1-pdg.dot']
    astFiles = ['0-ast.dot','1-ast.dot']

    # get all pesudo sub dirs
    cmd = 'find ' + dotDirPath + ' -maxdepth 1 -type d'
    print(cmd)
    proc = run(cmd, capture_output=True, shell=True)
    # 1st element is the pdgDir; the last one is '\n'
    subFolderPaths = proc.stdout.decode('utf-8').split('\n')[1:-1]
    subFolderPaths = sorted(subFolderPaths)
    print("there are {} subFolderPaths".format(len(subFolderPaths)))
    # print(subFolderPaths)

    for cnt, slicePath in enumerate(subFolderPaths):
        astPath = join(slicePath, 'ast')
        pdgPath = join(slicePath, 'pdg')
        astFilePath = join(astPath, astFiles[1])
        pdgFilePath = join(pdgPath, pdgFiles[1])
        if not exists(astFilePath):
            astFilePath = join(astPath, astFiles[0])
        if not exists(pdgFilePath):
            pdgFilePath = join(pdgPath, pdgFiles[0])
        print("working : {}".format(slicePath))
        # print("ast file {}".format(astFilePath))
        # print("pdg file {}".format(pdgFilePath))
        g_ast = nx.drawing.nx_agraph.read_dot(astFilePath)
        g_pdg = nx.drawing.nx_agraph.read_dot(pdgFilePath)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', "--datadir", type=str, help="dir to store data files")
    parser.add_argument('-n', "--dataname", type=str, help="name for data set")
    parser.add_argument('-dot', "--dotpath", type=str, help="the path of 'dot dir contains ASTs", default=None)
    args = parser.parse_args()
    if args.dotpath != None:
        # createDataset(args)
        chkDotCorret(args)
