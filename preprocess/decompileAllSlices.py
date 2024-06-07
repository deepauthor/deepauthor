from subprocess import run
import concurrent.futures
import os, re
import argparse
import angr
from subprocess import run, TimeoutExpired
from concurrent.futures import ProcessPoolExecutor

parser = argparse.ArgumentParser()
parser.add_argument('-d', "--directory", type=str, help="choose a slices' folder", required=True)
parser.add_argument('-e', "--errdir", type=str, help="a path for saving error log info", default="errlogs")
parser.add_argument("--multiyears", help="if mixed different year's gcj, default=True", action='store_false', dest='multiyears')
parser.add_argument('-n', "--numb", type=int, help="n parts of dataset", default=1)
parser.add_argument('-i', "--index", type=int, help="index of dataset", default=0)
parser.set_defaults(multiyears=True)
args = parser.parse_args()


def runOneTime(errdir, cmd):
    # cmd = cmd.split(' ')
    # cmd is in the form of cmdbase+file-path
    # file = cmd.split(' ')[-1]
    file = cmd.split('-f ')[-1]
    try:
         process = run(cmd, capture_output=True, timeout = 300, shell=True)
    except TimeoutExpired:
        return file
    else:
        if process.returncode != 0:
            logfile = os.path.join(errdir, os.path.basename(file))
            logfd = open(logfile+'.log', 'w+')
            logfd.write(process.stdout.decode("utf-8"))
            logfd.write(process.stderr.decode("utf-8"))
            logfd.close()
            return file
        else:
            # logfd.close()
            return None

if __name__ == '__main__':
    failed = []
    folder = args.directory
    # files = os.listdir(folder)
    # files = [os.path.join(folder,f) for f in files]
    # print(len(files))
    # logdir = os.path.join(folder, args.logDir)
    # if not os.path.exists(logdir):
    #       run(['mkdir', '-p', logdir])
    # files = [f for f in files if os.path.isfile(f) \
    #      and re.match("^\\.\\.\\/(\\w+\\/)+[A-Fa-f0-9]+$",f) \
    #      ]
    # print(len(files))
    # partLen = len(files)// args.numb
    # if args.index == args.numb -1:
    #     files = files[partLen*args.index :]
    # else:
    #     files = files[partLen*args.index: partLen*(args.index+1)]

    # pick out all slice files
    cmd = "find "+folder+" -type f -iname *.slice"
    res = run(cmd.split(" "),  capture_output=True, timeout = 60)
    sliceFiles = [e for e in res.stdout.decode("utf-8").split("\n") if e != '']
    # print(sliceFiles)

    # split all slices files into several parts
    partLen = len(sliceFiles)// args.numb
    if args.index == args.numb -1:
        sliceFiles = sliceFiles[partLen*args.index :]
    else:
        sliceFiles = sliceFiles[partLen*args.index: partLen*(args.index+1)]

    # to prepare a folder to save error logs
    errlogs = args.errdir
    if not os.path.exists(errlogs):
        print("create {} folder for error log files".format(args.errdir))
        run(("mkdir -p "+errlogs).split(" "))

    # authors' name for multi years are in the form of 'year_author'
    if args.multiyears:
        multiyears = " --multiyears "
    else:
        multiyears = ""
    cmdBase = "python3 decompileSlice.py "+multiyears+" -f "
    for sliceFile in sliceFiles:
        # sliceFile may contain spaces
        cmd = cmdBase + '"'+sliceFile+'"'
        rlt = runOneTime(errlogs, cmd)
        if rlt == sliceFile:
            print("{} timeout or error".format(sliceFile))
        else:
            print("{} terminate.".format(sliceFile))
