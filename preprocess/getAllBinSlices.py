from subprocess import run
import concurrent.futures
import os, re
import argparse
import angr
from subprocess import run, TimeoutExpired
from concurrent.futures import ProcessPoolExecutor
from os.path import join, dirname, abspath

parser = argparse.ArgumentParser()
parser.add_argument('-d', "--directory", type=str, help="path of a folder contains all bin files", required=True)
parser.add_argument('-l', "--logDir", type=str, help="log folder name, located the same parent dir as '-d'", default="bslogs")
parser.add_argument('-n', "--numb", type=int, help="n parts of dataset", default=1)
parser.add_argument('-i', "--index", type=int, help="index of dataset", default=0)
# parser.add_argument('-r', "--runcmd", type=str, help="cmd to run", required=True)

args = parser.parse_args()

def runOneTime(logdir, cmd):
    cmd = cmd.split(' ')
    file = cmd[-1]
    try:
         process = run(cmd, capture_output=True, timeout = 300)
    except TimeoutExpired:
        return file
    else:
        if process.returncode != 0:
            logfile = os.path.join(logdir, os.path.basename(file))
            logfd = open(logfile+'.log', 'w+')
            logfd.write(process.stdout.decode("utf-8"))
            logfd.write(process.stderr.decode("utf-8"))
            logfd.close()
            return file
        else:
            # logfd.close()
            return None

if __name__ == '__main__':
    binDir = args.directory
    cmd = 'find '+binDir +' -type f -executable'
    process = run(cmd, capture_output=True, shell=True)
    binFiles = process.stdout.decode('utf-8').split('\n')[:-1]
    # files = os.listdir(folder)
    # files = [os.path.join(folder,f) for f in files]
    # print(len(files))
    logdir = os.path.join(dirname(abspath(binDir)), args.logDir)
    print(logdir)
    if not os.path.exists(logdir):
          run(['mkdir', '-p', logdir])
    # files = [f for f in files if os.path.isfile(f) \
    #      and re.match("^\\.\\.\\/(\\w+\\/)+[A-Fa-f0-9]+$",f) \
    #      ]
    print(len(binFiles))
    # s = input()
    partLen = len(binFiles)// args.numb
    if args.index == args.numb -1:
        files = binFiles[partLen*args.index :]
    else:
        files = binFiles[partLen*args.index: partLen*(args.index+1)]

    cmdBase = "python3 sliceBinary.py -f "
    for file in files:
        cmd = cmdBase + file
        rlt = runOneTime(logdir, cmd)
        if rlt == file:
            print("{} timeout or error".format(file))
        else:
            print("{} terminate.".format(file))
