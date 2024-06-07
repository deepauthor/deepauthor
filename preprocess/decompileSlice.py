import angr
import os, ast, json
import argparse
from subprocess import run, TimeoutExpired
from os.path import join, dirname, basename, exists, splitext


def decompileSlice(args):
    # # pick out all slice files
    # folder = args.directionary
    # cmd = "find "+folder+" -type f -iname *.slice"
    # res = run(cmd.split(" "),  capture_output=True, timeout = 60)
    # sliceFiles = [e for e in res.stdout.decode("utf-8").split("\n") if e != '']
    # print(sliceFiles)
    sliceFile = args.file
    # all slice files at 'slice', but pesudo code file at "../pesudo"
    pesudo_path = join(dirname(dirname(sliceFile)), 'pesudo')
    if not exists(pesudo_path):
        os.makedirs(pesudo_path)

    with open(sliceFile, 'r') as f:
        data = json.load(f)
    chosenStmt = {eval(k):set(v) for k, v in data.items()}

    # bianry name contains: author, binaryName
    # slice file name contain: 'rest'/(api, argInd,), fAddr, fname, binary name;
    # fname is in the form of 'L+V'; 
    fileNameParts = splitext(basename(sliceFile))[0].split('_')
    if fileNameParts[0] == 'rest':
        start = 1
    else:
        start = 2
    start += 1
    fnAddr = fileNameParts[start-1]
    L = int(fileNameParts[start])
    start = start+1+L
    binName = "_".join(fileNameParts[start:])
    # print(binName)
    # fnAddr = basename(sliceFile).split('.')[0].split('_')[1]
    # if not args.multiyears:
    #     binName = fileNameParts[-1]

    binDir = os.path.dirname(os.path.dirname(sliceFile))
    binFile = os.path.join(binDir, binName)
    # print(binFile)

    p = angr.Project(binFile, load_options={"auto_load_libs": False})

    # get fn Obj from fn name
    cfgDecom = p.analyses.CFGFast(data_references=True, normalize=True)

    # func = cfgBS.kb.functions.function(name=fnNm)
    fns = [f for f in cfgDecom.kb.functions.values() \
     if f.binary_name == binName and not f.alignment \
      and f.name != ('sub_%x' % f.addr) \
     and not f.is_plt and f.size > 300]
    # print(chosenStmt)
    # fns += [cfgDecom.kb.functions.function(name="main")]
    for fn in fns:
        if fn.addr != int(fnAddr):
            continue
        # to do backward slicing on fn level
        cfgBS = p.analyses.CFGEmulated(keep_state=True, \
                                  state_add_options=angr.sim_options.refs, \
                                  starts = [fn.addr], \
                                  normalize = True, \
                                  call_depth = 0, \
                                  context_sensitivity_level=0)
        # cdg = p.analyses.CDG(cfgBS, start = fn.addr)
        # ddg = p.analyses.DDG(cfgBS, start = fn.addr)

        cfgDecom = p.analyses.CFGFast(data_references=True, normalize=True)
        # print(type(fn), type(chosenStmt))

        dec = p.analyses.Decompiler((fn,chosenStmt), cfg=cfgBS.model)

        code = dec.codegen.text
        # print(code)

        pesudoCodeFiles = join(pesudo_path , basename(sliceFile)[:-len(".slice")]+".c")
        # print(sliceFile)
        # print(pesudoCodeFiles)
        with open(pesudoCodeFiles, 'w') as pcf:
            pcf.write(code)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', "--file", type=str, help="a slice file", required=True)
    parser.add_argument("--multiyears", help="switch for checking if the dataset is over years or not, default=False", action='store_true', dest='multiyears')
    parser.set_defaults(multiyears=False)
    args = parser.parse_args()
    decompileSlice(args)