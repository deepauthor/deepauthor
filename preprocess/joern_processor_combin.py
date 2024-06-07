import subprocess, os, argparse
from os.path import join, abspath, basename, exists, splitext

CD = 'cd /home/yihua/joern/bin/joern-cli'
# JOERN = './joern --script /Users/py/test.sc --params cpgFile=/Users/py/pesudo_c/00000000002a00a0,projectName=00000000002a00a0'
# AST = '/home/yihua/Desktop/joern/dot/ast'
# PDG = '/home/yihua/Desktop/joern/dot/pdg'
# candi = [
#         '/home/yihua/Desktop/projects/AuthorshipIdentification/gcj2020/binTest/sliceFunctions/slices', \
#         '/home/yihua/Desktop/projects/AuthorshipIdentification/angrInstallTest/codeWithNewAngr/mytest/slices', \
#         '/home/yihua/Desktop/projects/AuthorshipIdentification/gcj2020/binTest/caseFromC/slices', \
#         '/Users/py/pesudo_c', \
#         ]
# pesudo_c_path = candi[2]
CPG_DIR = '/home/yihua/joern/bin/joern-cli'
# importCode(inputPath="/Users/py/pesudo_c/00000000002a00a0", projectName="00000000002a00a0")
# run.ossdataflow
# save
# joern-export --repr ast --out 00000000002a00a0
# joern-export --repr pdg --out 00000000002a00a0
parser = argparse.ArgumentParser()
parser.add_argument('-n', "--number", type=int, help="slice dataset into number parts", default=1)
parser.add_argument('-i', "--index", type=int, help="index of the part of dataset", default=0)
parser.add_argument('-p', "--psdcpath", type=str, help="the path of all pesudo c files", required=True)
parser.add_argument('-dot', "--dotpath", type=str, help="the path of 'dot' dir contains all ASTs & PDGs dot files", default=0)
args = parser.parse_args()
pesudo_c_path=os.path.abspath(args.psdcpath)
# DOT = args.dotpath
# AST = os.path.join(os.path.abspath(args.dotpath), 'ast')
# if not os.path.exists(AST):
#     os.makedirs(AST)
# PDG = os.path.join(os.path.abspath(args.dotpath), 'pdg')
# if not os.path.exists(PDG):
#     os.makedirs(PDG)

# Using scala script instead of interactive scripts
# ./joern --script /Users/py/test.sc --params cpgFile=/Users/py/pesudo_c/00000000002a00a0,projectName=00000000002a00a0

numb = args.number
nthPart = args.index
if numb > 1:
    for x in range(numb):
        cpgDir = 'thread_'+str(nthPart)
        CPG_PATH = os.path.join(CPG_DIR, cpgDir)
        if not os.path.exists(CPG_PATH):
            cmd = 'mkdir -p '+CPG_PATH
            subprocess.run(cmd.split(' '))
else:
    # using default path
    CPG_PATH = CPG_DIR

# get all pesudo folders
cmd = "find "+pesudo_c_path+" -type f -name '*.c' "
proc = subprocess.run(cmd, shell=True, capture_output = True)
pesudo_c_files = proc.stdout.decode('utf-8').split('\n')[:-1]

filenames = pesudo_c_files
partLen = len(filenames) // numb
if numb == (nthPart + 1) and numb > 1:
    filenames = filenames[partLen * nthPart :]
else:
    filenames = filenames[(partLen * nthPart) : (partLen * (nthPart + 1))]

# create a dir for each file
for file in filenames:
    dot_file_dir = join(args.dotpath, splitext(basename(file))[0])
    if not exists(dot_file_dir):
        os.makedirs(dot_file_dir)
    else:
        continue
# get ast and pdg for each file
for i, file in enumerate(filenames, start=1):
    # name = os.path.splitext(file)[0]
    print("working on {}".format(file))
    name = splitext(basename(file))[0]
    dotpath = abspath(args.dotpath)

    PARSE = './joern-parse -o ' + os.path.join(CPG_PATH, 'cpg.bin') + ' '+ '"'+file+'"'
    PARSE_SAVE = CD + ' && ' + PARSE
    print(PARSE_SAVE)
    subprocess.run(PARSE_SAVE, shell=True)

    # each file has ast and pdg dir
    AST_OUT_DOT =  dotpath+ '/'+ name + '/ast'
    AST_DOT_CMD = CD + ' && ./joern-export ' + os.path.join(CPG_PATH, 'cpg.bin') +' --repr ast --out ' + AST_OUT_DOT
    print(AST_DOT_CMD)
    subprocess.run(AST_DOT_CMD, shell=True)

    PDG_OUT_DOT = dotpath+ '/'+ name + '/pdg'
    PDG_DOT_CMD = CD + ' && ./joern-export ' + os.path.join(CPG_PATH, 'cpg.bin') +' --repr pdg --out ' + PDG_OUT_DOT
    print(PDG_DOT_CMD)
    subprocess.run(PDG_DOT_CMD, shell=True)
    # if i == 500:
    print(("="*20)+"\t processed [{}/{}] ".format(i, partLen)+("="*20))

# remove temp 'thread_x' folder
if numb > 1:
    if os.path.exists(CPG_PATH):
        cmd = 'rm -r '+CPG_PATH
        subprocess.run(cmd, shell=True)

# generateDotFileFromParse()