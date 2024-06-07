import angr, argparse, pyvex
import os, json
from subprocess import run, TimeoutExpired
from os.path import dirname, basename

# parser = argparse.ArgumentParser()
# parser.add_argument('-f', "--file", type=str, help="choose file", required=True)
# args = parser.parse_args()

def sliceBinary(args):
    cwd = os.getcwd()
    fileNm = args.file
    # in case of the path of the file is a related path
    fileNm = os.path.join(cwd, fileNm)
    print(fileNm)
    author = basename(dirname(fileNm))

    # read arguments types of all APIs
    apiArgDict = {}
    with open("apiArgs.txt", "r") as f:
        data = f.readlines()
        for l in data:
            x = l.split(":")
            apiArgDict.update({x[0]:x[1].replace("\n","").replace(" ", "").split(",")})

    p = angr.Project(fileNm, load_options={"auto_load_libs": False})
    cfgDecom = p.analyses.CFGFast(data_references=True, normalize=True)

    idfer = p.analyses.Identifier(cfg=cfgDecom)
    addr2NameDict = {}
    for funcInfo in idfer.func_info:
        print(funcInfo.demangled_name)
        # addr2NameDict.update({hex(funcInfo.addr): funcInfo.name})
        if 'std::' in funcInfo.demangled_name:
            # this branch is for C++ code
            # the demangled_name is in the form of 'xx std::yy(zz)' xx, yy, and zz are complex
            # yy could be function name ; zz may contain many 'std::'
            if funcInfo.demangled_name.split('std::',1)[1].split('(')[0] in apiArgDict.keys():
                addr2NameDict.update({hex(funcInfo.addr): funcInfo.demangled_name[len('std::'):].split('(')[0]})
        else:
            # this branch is for C code
            addr2NameDict.update({hex(funcInfo.addr): funcInfo.name})
    if len(addr2NameDict.keys())== 0:
        print("empty addr2NameDict")
    """
    The first six integer or pointer arguments are passed in registers RDI, RSI, RDX, RCX, R8, R9 
    (R10 is used as a static chain pointer in case of nested functions
    while XMM0, XMM1, XMM2, XMM3, XMM4, XMM5, XMM6 and XMM7 are used for the first floating point arguments.
    """
    intPtArgs = [["RDI","rdi"],["RSI","rsi"],["RDX","rdx"],["RCX","rcx"],["R8","r8"],["R9","r9"],["R10","r10"]]
    fltArgs = [["XMM0","xmm0"],["XMM1","xmm1"],["XMM2","xmm2"],["XMM3","xmm3"],["XMM4","xmm4"],["XMM5","xmm5"],["XMM6","xmm6"],["XMM7","xmm7"]]

    binary_name = os.path.basename(fileNm)
    fns = [f for f in cfgDecom.kb.functions.values() \
     if f.binary_name == binary_name and not f.alignment \
     and f.name != ('sub_%x' % f.addr) \
     and not f.is_plt and f.size > 300]

    sliceDir = os.path.join(os.path.dirname(fileNm), 'slices')
    if not os.path.exists(sliceDir):
          run(['mkdir', '-p', sliceDir])

    f_index = 0
    for func in fns:
        # print("working for {}, demangled_name: {}".format(func.name, func.demangled_name))
        # continue
        # demangled_name in the form of "fn(type, type1, ...)" except 'main'
        # demangled_name can start with "_Z"
        if 'std::' in func.demangled_name:
            # this branch is for C++ code
            # c++ function prototype form: xx std::yy(zz)
            if func.demangled_name.split('std::',1)[1].split('(')[0] in apiArgDict.keys():
                # yy could be function name or function template
                fname = func.demangled_name.split('std::',1)[1].split('(')[0]
            else:
                # this branch is for the case of complexed C++ prototype
                fname = 'cppFn'+str(f_index)
                f_index += 1
        else:
            # because of bug #13 for the case '2019_Dormi_0000000000106d06'
            if func.demangled_name[:2] == '_Z':
                fname = 'cppFn'+str(f_index)
                f_index += 1
            else:
                fname = func.demangled_name.split('(')[0] 
        # use len+v format to keep func name
        fname = str(len(fname.split('_')))+"_"+fname

        # to get function level' CFG, CDG, DDG for backward slicing
        cfgBS = p.analyses.CFGEmulated(keep_state=True, \
                                  state_add_options=angr.sim_options.refs, \
                                  starts = [func.addr], \
                                  normalize = True, \
                                  call_depth = 0, \
                                  context_sensitivity_level=0)
        cdg = p.analyses.CDG(cfgBS, start = func.addr)
        ddg = p.analyses.DDG(cfgBS, start = func.addr)

        # to slice a function from each API call site 
        callblks = [blk for blk in func.blocks if blk.vex.jumpkind == "Ijk_Call"]
        slices = []
        for blk in callblks:
            # last ins of blk should call an address(a function)
            ins = blk.capstone.insns[-1]
            target = ins.op_str
            stmts = blk.vex.statements

            # get API name
            apiName = addr2NameDict.get(target)

            # get all arguments of the API
            args = apiArgDict.get(apiName)
            if args is None:
                print("skip target {}".format(apiName))
                continue
            target_node = cfgBS.model.get_any_node(blk.addr)
            if target_node == None:
                print("address conflict")
                continue

            # to do backward slice for each arg
            print("target {} with args {}".format(apiName, args))
            intPtIndex = 0
            fltIndex = 0
            for arg in args:
                # to find location of an arg in IR in the following loop
                start = len(stmts)
                bs_flag = False
                for i in range(start, 0, -1):
                    # IR used PUT to assign value to specific reg.s
                    if not isinstance(stmts[i-1] , pyvex.stmt.Put):
                        continue
                    stmt = stmts[i-1].__str__(arch=blk.vex.arch, tyenv=blk.vex.tyenv)
                    if arg == 'int' or arg == 'pointer':
                        if intPtArgs[intPtIndex][0] in stmt \
                            or intPtArgs[intPtIndex][1] in stmt:
                            start = i-1
                            intPtIndex += 1
                            bs_flag = True
                            break
                    elif arg == 'float':
                        if fltArgs[fltIndex][0] in stmt \
                            or fltArgs[fltIndex][1] in stmt:
                            start = i-1
                            fltIndex += 1
                            bs_flag = True
                            break
                    else:
                        # arg could be 'skip_pointer' or other types, then ignore.
                        start -= 1
                        bs_flag = False
                        continue
                if bs_flag:
                    print("blk address {}; start {}".format(hex(blk.addr), start))
                    bs = p.analyses.BackwardSlice(cfgBS, cdg=cdg, ddg=ddg, targets=[ (target_node, start) ])
                    slices.append({'api':apiName, 'argIndx': args.index(arg), 'type': arg, 'bs':bs})
                    print(bs.chosen_statements)

        print("how many slices are there? {}".format(len(slices)))

        # log all slices into a separated file
        # merge all statements in all slices for getting the rest of IRs
        # binary_name : author_name + binary
        # name format: api_argindex_funcName_funcAddr_binaryName
        # funcAddr is used for the process of decompilation
        slicedStmts = {}
        restStmts = {}
        for bs in slices:
            # sliceName = [bs.get('api'),str(bs.get('argIndx')),bs.get('type'),fname, func.binary_name]
            sliceName = [bs.get('api'),str(bs.get('argIndx')), str(func.addr), fname, func.binary_name]
            sliceFileName = '_'.join(sliceName)+'.slice'
            sliceFileName = os.path.join(sliceDir,sliceFileName)
            print(sliceFileName)
            fmtDict = bs.get('bs').chosen_statements
            fmtDict = {k: list(v) for k, v in fmtDict.items()}
            with open(sliceFileName, 'w') as sf:
                json.dump(fmtDict, sf, indent=4)
            
            for k, v in bs.get('bs').chosen_statements.items():
                if k not in slicedStmts.keys():
                    slicedStmts.update({k:v})
                else:
                    slicedStmts[k] = slicedStmts.get(k).union(v)
        print(slicedStmts)

        # the rest IR statements after slicing
        # for blk in func.blocks:
        #     blk.vex.pp()
        #     blk.pp()
        for blk in func.blocks:
            if blk.addr in slicedStmts.keys():
                restStmts[blk.addr] = set([i for i in range(0, len(blk.vex.statements)) \
                                        if i not in slicedStmts[blk.addr]])
            else:
                restStmts[blk.addr] = set([i for i in range(0, len(blk.vex.statements))]+[-2])
        print(restStmts)
        # added fname for function level on May 6
        # print(["rest", fname, str(func.addr), author, func.binary_name])
        sliceFileName = '_'.join(["rest", str(func.addr), fname, func.binary_name])+'.slice'
        sliceFileName = os.path.join(sliceDir,sliceFileName)
        print(sliceFileName)
        fmtDict = {k:list(v) for k, v in restStmts.items()}
        with open(sliceFileName, 'w') as sf:
            json.dump(fmtDict, sf, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', "--file", type=str, help="choose file", required=True)
    args = parser.parse_args()
    sliceBinary(args)