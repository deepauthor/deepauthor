from subprocess import run
from os.path import exists, basename, join, splitext, dirname
import os, argparse, re
import faulthandler

faulthandler.enable()

# cmd = 'python3 getASTorPDG_combin.py -d '+args.datadir+" -n "+args.dataname+' -dot '+args.dotpath
# p = run(cmd, shell=True)
# if p.returncode == 0:
#     print("getASTorPDG_combin.py finished")
# else:
#     print("something wrong at the last step")
def testByAuthor(author, args, tmp_path, tmp_data):
    print("remove fn idx: {}".format(args.indx))
    test = tmp_path+'/test'
    cmd = "find "+args.dotpath+" -maxdepth 1 -type d -iname '*"+author+"*'"
    p = run(cmd, capture_output=True, shell=True)
    dirs = p.stdout.decode('utf-8').split('\n')[:-1]
    print("{} has {} dirs".format(author, len(dirs)))
    for e in dirs:
        if ' ' in e:
            e = '"'+e+'"'
        cmd = 'cp -r '+e+' '+tmp_path
        # print(cmd)
        run(cmd, shell=True, capture_output=True)
    print("copied to {}".format(tmp_path))
    run('rm -r '+test, shell=True)

    # testErr = 'python3 getASTorPDG_combin.py -d '+tmp_data+" -n "+args.dataname+' -dot '+tmp_path
    # testErr = 'python3 verify_dot.py -d '+tmp_data+" -n "+args.dataname+' -dot '+tmp_path
    errlog_dir = join(dirname(args.dotpath), "dot_err_logs")
    if not exists(errlog_dir):
        run("mkdir "+errlog_dir, shell=True)
    errlog_file = join(errlog_dir, "_".join([args.indx, args.errlog]))
    errlog_file_p = open(errlog_file, 'w')

    testErr = 'python3 verify_dot.py -dot '+tmp_path
    proc = run(testErr, capture_output=True, shell=True)
    times = 1
    print(testErr)
    print(proc.stdout.decode('utf-8'))
    print(proc.stderr.decode('utf-8'))
    while proc.returncode != 0:
        errlog_file_p.write("met errors: {} times\n".format(times))
        dirs = [s[len('working : '):] for s in proc.stdout.decode('utf-8').split('\n')[:-1] if re.match(r'^working : ((\/\w*)*)', s)]
        # print(dirs)
        for d in dirs:
            if ' ' in d:
                d = '"'+d+'"'
            delDirs = 'rm -r '+d
            run(delDirs, shell=True)

        print(dirs)
        if ' ' in basename(dirs[-1]):
            delOrgDir = 'rm -r '+'"'+join(args.dotpath, basename(dirs[-1]))+'"'
        else:
            delOrgDir = 'rm -r '+join(args.dotpath, basename(dirs[-1]))
        print(delOrgDir)
        run(delOrgDir, shell=True)
        p = run('find '+tmp_path+' -type d', shell=True, capture_output=True)
        numOfRemain = len(p.stdout.decode('utf-8').split('\n')[1:-1])
        if numOfRemain==0:
            print('no dir remaining')
            times += 1
            break
        else:
            proc = run(testErr, capture_output=True, shell=True)
            print(testErr)
            print(proc.stdout.decode('utf-8'))
            print(proc.stderr.decode('utf-8'))
            times += 1
    errlog_file_p.write("met errors: {} times\n".format(times - 1))
    errlog_file_p.close()
    cmd = 'rm -r '+tmp_path+' && mkdir -p '+test
    run(cmd, shell=True)

def main(args):
    # create temp folders for testing AST & PDG
    tmp_path =  '../tmp'+args.indx
    tmp_data = '../data'+args.indx
    test = tmp_path+'/test'
    if not exists(tmp_path):
        os.makedirs(tmp_path)
    else:
        run('rm -r '+tmp_path, shell=True)

    if not exists(tmp_data):
        os.makedirs(tmp_data)
    else:
        run('rm -r '+tmp_data, shell=True)
    # when temp folder is empty, exists error while cp dir to the folder 
    # so we make a test folder to void the error
    run('mkdir -p '+test, shell=True)
    cmd = 'find '+args.dotpath+'  -maxdepth 1 -mindepth 1 -type d'
    p = run(cmd, shell=True, capture_output=True)
    dirs = p.stdout.decode('utf-8').strip().split('\n')
    # authors = ['_'.join(basename(d).split('_')[2:-1]) for d in dirs]
    authors = [getAuthorFromPath(d) for d in dirs]
    authors = list(set(authors))
    authors = [e for e in authors if e!='']
    authors.sort()
    print("there are {} authors: {}".format(len(authors), authors))
    # print(authors)
    partlen = len(authors)//args.parts
    indx = int(args.indx)
    if indx + 1 >= args.parts:
        parts = authors[(partlen * indx ):]
    else:
        parts = authors[(partlen * indx):(partlen*(indx + 1))]
    for author in parts:
        # if i !=0:
        #     conintue
        testByAuthor(author, args, tmp_path, tmp_data)
    cmd = 'rm -r '+tmp_path+' '+tmp_data
    run(cmd, shell=True)
    print("test finished")

def getAuthorFromPath(dirname):
    # "rest(api_argInde)_fname_faddr_biname"
    # fname: L+V
    # biname = round_id + challenge_id + 
    start = 0
    dirNameParts = basename(dirname)[:-len("-optx")].split("_")
    author = dirNameParts[-1]
    return author

def getAuthorFrom(fileBase):
    sliceNameParts = basename(fileBase).split('_')
    start = 0
    if 'rest'==sliceNameParts[start]:
        start = 1
    else:
        start = 2
    L = int(sliceNameParts[start])
    start = start+1+L+1
    author = '_'.join(sliceNameParts[start:-1])
    return author

def test4author(args):
    # create temp folders for testing AST & PDG
    if args.indx == None:
        indx = ''
    tmp_path =  '../tmp'+indx
    tmp_data = '../data'+indx
    test = tmp_path+'/test'
    if not exists(tmp_path):
        os.makedirs(tmp_path)
    else:
        run('rm -r '+tmp_path, shell=True)
    if not exists(tmp_data):
        os.makedirs(tmp_data)
    else:
        run('rm -r '+tmp_data, shell=True)
    # when temp folder is empty, exists error while cp dir to the folder 
    # so we make a test folder to void the error
    run('mkdir -p '+test, shell=True)
    testByAuthor(args.author, args, tmp_path, tmp_data)
    cmd = 'rm -r '+tmp_path+' '+tmp_data
    run(cmd, shell=True)
    print("over")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('-d', "--datadir", type=str, help="to show number of binaries for top N", required=True)
    parser.add_argument('-n', "--dataname", type=str, help="name for data set")
    parser.add_argument('-dot', "--dotpath", type=str, help="the path of 'dot dir contains ASTs", default=None)
    parser.add_argument('-p', "--parts", type=int, help="the number of parts", default=1)
    parser.add_argument('-i', "--indx", type=str, help="index for this thread", default=None)
    parser.add_argument('-a', "--author", type=str, help='author name', default=None)
    parser.add_argument('-e', "--errlog", type=str, help='file path for recording process', default="dotErr.log")

    args = parser.parse_args()
    if args.author != None:
        test4author(args)
    else:
        main(args)
    # test(args)