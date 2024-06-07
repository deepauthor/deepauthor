import os, argparse, json, random
import networkx as nx
import numpy as np
from subprocess import run
import torch
from unixcoder import UniXcoder
from os.path import exists, join, basename, splitext, abspath, isdir, dirname
import os, argparse, re
from concurrent.futures import ThreadPoolExecutor
import shutil
import Util

class InputFileBase():
    """docstring for InputFileBase"""
    def __init__(self, args):
        self.args = args
        dotDirPath = os.path.abspath(args.dotpath)
        self.pdgFiles = ['0-pdg.dot','1-pdg.dot']
        self.astFiles = ['0-ast.dot','1-ast.dot']

        # get all pesudo sub dirs
        cmd = 'find ' + dotDirPath + ' -mindepth 1 -maxdepth 1 -type d'
        # print(cmd)
        proc = run(cmd, capture_output=True, shell=True)
                                                                
        subDirs = proc.stdout.decode('utf-8').strip().split('\n')
        self.sortedSlicePaths = sorted(subDirs)
        self.subDirsLen = len(subDirs)
        print("there are {} subdirs".format(self.subDirsLen))
        self.getAuthorsSet()
        
        with open('apiArgs.txt','r') as f:
            apiArgs = f.read().split('\n')
        self.apiList = [e.split(':')[0] for e in apiArgs]

        self.relationalOp = ['greaterThan', 'greaterEqualsThan', 'equals', 'notEquals', 'lessThan', 'lessEqualsThan']

        self.dataRoot = join(args.datadir, args.dataname)
        if not exists(self.dataRoot):
            os.makedirs(self.dataRoot)

    def getAuthorFromPath(self, dirname):
        # in the form of "rest(api_argInde)_faddr_fname_biname", where
        # fname : L+V; biname: year_author_biname
        start = 0
        dirNameParts = basename(dirname).split("_")
        if dirNameParts[start] == 'rest':
            start = 1
        else:
            start = 2
        L = int(dirNameParts[start])
        start += 1
        start = start+1+L
        author = "_".join(dirNameParts[start:-1])
        return author

    def getAuthorsSet(self ):
        # get all authors for all slice files from the name of folders

        authors = []
        for dirname in self.sortedSlicePaths:
            if dirname == '':
                continue
            author = self.getAuthorFromPath(dirname)
            authors.append(author)
        tmp = set(authors)

        self.authors = sorted(list(tmp))
        print("there are {} authors".format(len(self.authors)))

class DiffInput(InputFileBase):
    """docstring for DiffInput
        to create 10 files in total as input files
        a folder for AST and a folder for PDG
    """
    def __init__(self, args):
        super(DiffInput, self).__init__(args)

        # prepare cooperating paths and prefixes 
        dataRoot = self.dataRoot
        astDataRoot = join(dataRoot, args.dataname+'_ast')
        pdgDataRoot = join(dataRoot, args.dataname+'_pdg')
        if not exists(dataRoot):
            os.makedirs(astDataRoot)
            os.makedirs(pdgDataRoot)
        # prefix = join(args.datadir, args.dataname, args.dataname)
        self.astPrefix = join(astDataRoot,basename(astDataRoot))
        self.pdgPrefix = join(pdgDataRoot,basename(pdgDataRoot))
        print(self.dataRoot, self.astPrefix, self.pdgPrefix)

        
    def mkRecords2_A(self, g, nodeId, aFile, cnt, end):
        adjMatrix = nx.linalg.graphmatrix.adjacency_matrix(g, weight='None').toarray()
        positions = np.array(np.where(adjMatrix==1)).T
        rows = positions.shape[0]
        positions += nodeId
        for i in range(rows):
            # write each position to DS_A.txt
            row, col = tuple(positions[i].tolist())
            aFile.write("{}, {}".format(row, col))
            if not ((cnt == end) and (i == rows -1)):
                aFile.write('\n')
        return nodeId + adjMatrix.shape[0], adjMatrix.shape[0]

    def createDataset(self):
        # delete exists target files
        cmd = 'find ' + self.dataRoot + ' -type f'
        proc = run(cmd.split(' '), capture_output=True)
        files = proc.stdout.decode('utf-8').split('\n')[:-1]
        print("exists {} data files".format(len(files)))
        if len(files) > 0:
            run(['rm']+files)
            print("deleted existed files. Done.")
        # 5 files will be recorded, open them first as append mode
        aFile_ast = open(self.astPrefix + '_A.txt',"a")
        giFile_ast = open(self.astPrefix + '_graph_indicator.txt', "a")
        glFile_ast = open(self.astPrefix + '_graph_labels.txt', "a")
        naFile_ast = open(self.astPrefix + '_node_attributes.txt', "a")
        nlFile_ast = open(self.astPrefix + '_node_labels.txt', "a")

        aFile_pdg = open(self.pdgPrefix + '_A.txt',"a")
        giFile_pdg = open(self.pdgPrefix + '_graph_indicator.txt', "a")
        glFile_pdg = open(self.pdgPrefix + '_graph_labels.txt', "a")
        naFile_pdg = open(self.pdgPrefix + '_node_attributes.txt', "a")
        nlFile_pdg = open(self.pdgPrefix + '_node_labels.txt', "a")

        # to add API name as an attribution for g
        gnFile_ast = open(self.astPrefix + '_graph_api.txt', "a")
        gnFile_pdg = open(self.pdgPrefix + '_graph_api.txt', "a")

        # # check API info for
        # with open('apiArgs.txt','r') as f:
        #     apiArgs = f.read().split('\n')
        # apiList = [e.split(':')[0] for e in apiArgs]

        # load pre-trained model for generating pdg nodes embedding
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = UniXcoder("microsoft/unixcoder-base-nine")
        model.to(device)

        # for ast nodes' embedding
        with open('nodeTypes.txt', 'r') as nt:
            nodeTypes = nt.read().split('\n')
        nodeTypes = sorted(nodeTypes)

        graphIndicator = 1
        nodeID_ast = 1
        nodeMax = 0
        nodeID_pdg = 1
        for cnt, sub in enumerate(self.sortedSlicePaths):
            astPath = join(sub, 'ast')
            pdgPath = join(sub, 'pdg')
            astFile = join(astPath, self.astFiles[1])
            pdgFile = join(pdgPath, self.pdgFiles[1])
            if not exists(astFile):
                astFile = join(astPath, self.astFiles[0])
            if not exists(pdgFile):
                pdgFile = join(pdgPath, self.pdgFiles[0])
            print("working : {}".format(sub))
            # print("ast file {}".format(astFile))
            # print("pdg file {}".format(pdgFile))
            g_ast = nx.drawing.nx_agraph.read_dot(astFile)
            g_pdg = nx.drawing.nx_agraph.read_dot(pdgFile)
            if g_ast.number_of_nodes() == 0 or g_pdg.number_of_nodes() == 0: # for any empty graph
                continue
            # 1. get the data of DS_A.txt
            nodeID_ast, adjshape0 = mkRecords2_A(g_ast,nodeID_ast, aFile_ast, cnt, self.subDirsLen-1)
            nodeID_pdg, _ = mkRecords2_A(g_pdg,nodeID_pdg, aFile_pdg, cnt, self.subDirsLen-1)
            # 5. for ast get the data of DS_node_labels.txt, assume only one class
            nodeMax += adjshape0
            # adjMatrix = nx.linalg.graphmatrix.adjacency_matrix(g, weight='None').toarray()
            # positions = np.array(np.where(adjMatrix==1)).T
            # rows = positions.shape[0]
            # positions += nodeID
            # for i in range(rows):
            #     # write each position to DS_A.txt
            #     row, col = tuple(positions[i].tolist())
            #     aFile.write("{}, {}".format(row, col))
            #     if not ((cnt == len(subDirs)-1) and (i == rows -1)):
            #         aFile.write('\n')
            # nodeID += adjMatrix.shape[0]

            # 3. get the data of DS_graph_labels.txt
            # to add graph corresponding API name for
            author = '_'.join(splitext(basename(sub))[0].split('_')[2:-1])
            label = self.authors.index(author)
            glFile_ast.write(str(label))
            glFile_pdg.write(str(label))
            if cnt != self.subDirsLen -1:
                glFile_ast.write("\n")
                glFile_pdg.write("\n")

            # pdg
            # 2. get the data of DS_graph_indicator.txt; each row is a gi of this nodeID
            # 4. get the data of DS_node_attributes.txt
            # 5. get the data of DS_node_labels.txt, assume four classes
            # 5. 1 - control ; 2 - statement ; 3 - func call ; 4 - others
            for i, node in enumerate(g_pdg.node):
                giFile_pdg.write(str(graphIndicator))

                # feat = np.zeros(len(PDGnodeTypes), dtype=int)
                # label form : "(<operator>.xxx, experess)"
                code = g_pdg.node.get(node).get('label').split(',')[1]
                # feat = getCodeEmbedding(code)
                tokens_ids = model.tokenize([code], max_length=512, mode="<encoder-only>")
                source_ids = torch.tensor(tokens_ids).to(device)
                _, code_embedding = model(source_ids)

                # feat = [str(e) for e in code_embedding]
                feat = code_embedding.squeeze().tolist()
                feat = [str(e) for e in feat]
                naFile_pdg.write(', '.join(feat))

                # check and write node label
                # depending on operator part to decide label
                first_part = g_pdg.node.get(node).get('label').split(',')[0]
                if 'operator' in first_part:
                    op = first_part.split('.')[1]
                    if op in self.relationalOp:
                        label = 1
                    else:
                        label = 2
                else:
                    if first_part in self.apiList:
                        label = 3
                    else:
                        label = 4
                nlFile_pdg.write(str(label))

                if not ((cnt == self.subDirsLen-1) and (i == len(g_pdg.node) -1)):
                    giFile_pdg.write('\n')
                    naFile_pdg.write('\n')
                    nlFile_pdg.write('\n')

            # ast
            # 2. get the data of DS_graph_indicator.txt; each row is a gi of this nodeID
            # 4. get the data of DS_node_attributes.txt
            for i, node in enumerate(g_ast.node):
                giFile_ast.write(str(graphIndicator))

                feat = np.zeros(len(nodeTypes), dtype=int)
                cut = g_ast.node.get(node).get('label').split(',')[0]
                if 'operator' in cut:
                    cut = 'operator'
                else:
                    cut = cut[1:]
                    if cut in self.apiList:
                        cut = 'API'
                    elif cut not in nodeTypes:
                        cut = 'UNKNOWN'
                    else:
                        pass
                feat[nodeTypes.index(cut)] = 1
                feat = feat.tolist()
                feat = [str(e) for e in feat]
                naFile_ast.write(', '.join(feat))
                if not ((cnt == self.subDirsLen-1) and (i == len(g_ast.node) -1)):
                    giFile_ast.write('\n')
                    naFile_ast.write('\n')

            print("{}/{} :{} processed".format(cnt, self.subDirsLen, sub))
            graphIndicator += 1

        # to close 5 files before exit
        print("authors: {}; graphs: {}".format(len(self.authors), graphIndicator))

        # write and save DS_node_labels.txt
        nodeLabelVector = ['1']*nodeMax
        nlFile_ast.write('\n'.join(nodeLabelVector))

        aFile_ast.close()
        giFile_ast.close()
        glFile_ast.close()
        naFile_ast.close()
        nlFile_ast.close()

        aFile_pdg.close()
        giFile_pdg.close()
        glFile_pdg.close()
        naFile_pdg.close()
        nlFile_pdg.close()
        print("createDataset over")

    def createSpecificFile(self, key):
        handle = {'nl_pdg':self.createNL,
                  'gn':self.createGN
                  }
        if key in handle.keys():
            handle[key]()
        else:
            print("{} is not supported".format(key))

    def createNL(self):
        print("only create nl_pdg file")
        nlFile_pdg = open(self.pdgPrefix + '_node_labels.txt', "a")
        for cnt, sub in enumerate(self.sortedSlicePaths):
            print('working on {}'.format(sub))
            pdgPath = join(sub, 'pdg')
            pdgFile = join(pdgPath, self.pdgFiles[1])
            if not exists(pdgFile):
                pdgFile = join(pdgPath, self.pdgFiles[0])
            g_pdg = nx.drawing.nx_agraph.read_dot(pdgFile)
            for i, node in enumerate(g_pdg.node):
                # check and write node label
                # depending on operator part to decide label
                first_part = g_pdg.node.get(node).get('label').split(',')[0]
                if 'operator' in first_part:
                    op = first_part.split('.')[1]
                    if op in self.relationalOp:
                        label = 1
                    else:
                        label = 2
                else:
                    if first_part in self.apiList:
                        label = 3
                    else:
                        label = 4
                nlFile_pdg.write(str(label))

                if not ((cnt == self.subDirsLen-1) and (i == len(g_pdg.node) -1)):
                    nlFile_pdg.write('\n')
            print("{}/{} :{} processed".format(cnt, self.subDirsLen, sub))
        nlFile_pdg.close()
        print("createSpecificFile over")

    def createGN(self):
        # delete exists target files
        cmd = 'find ' + self.dataRoot + ' -type f'
        proc = run(cmd.split(' '), capture_output=True)
        files = proc.stdout.decode('utf-8').split('\n')[:-1]
        print("exists {} data files".format(len(files)))
        # if len(files) > 0:
        #     run(['rm']+files)
        #     print("deleted existed files. Done.")
        # to add API name as an attribution for g
        gnFile_ast = open(self.astPrefix + '_graph_api.txt', "a")
        gnFile_pdg = open(self.pdgPrefix + '_graph_api.txt', "a")
        graphIndicator = 1
        nodeID_ast = 1
        nodeMax = 0
        nodeID_pdg = 1
        for cnt, sub in enumerate(self.sortedSlicePaths):
            astPath = join(sub, 'ast')
            pdgPath = join(sub, 'pdg')
            astFile = join(astPath, self.astFiles[1])
            pdgFile = join(pdgPath, self.pdgFiles[1])
            if not exists(astFile):
                astFile = join(astPath, self.astFiles[0])
            if not exists(pdgFile):
                pdgFile = join(pdgPath, self.pdgFiles[0])
            print("working : {}".format(sub))
            # print("ast file {}".format(astFile))
            # print("pdg file {}".format(pdgFile))
            g_ast = nx.drawing.nx_agraph.read_dot(astFile)
            g_pdg = nx.drawing.nx_agraph.read_dot(pdgFile)
            if g_ast.number_of_nodes() == 0 or g_pdg.number_of_nodes() == 0: # for any empty graph
                continue
            api = '_'.join(splitext(basename(sub))[0].split('_')[2:-1])
            label = self.authors.index(author)
            glFile_ast.write(str(label))
            glFile_pdg.write(str(label))
            if cnt != self.subDirsLen -1:
                glFile_ast.write("\n")
                glFile_pdg.write("\n")

class ModifiedDiffInput(InputFileBase):
    """ docstring for DiffModelInput
        to create input files for our model in args.datadir
        This Class will create a set of folders from all slices. 
        Each created folder represents a slice, which contains
        'gpickle' files for AST & PDG.
    """
    def __init__(self, args):
        super(ModifiedDiffInput, self).__init__(args)

        self.graphpostfix = '.gpickle'

        # load pre-trained model for generating pdg nodes embedding
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = UniXcoder("microsoft/unixcoder-base-nine")
        self.model.to(self.device)

        # ast nodes' embedding will be in the form of one-hot coding
        with open('nodeTypes.txt', 'r') as nt:
            nodeTypes = nt.read().split('\n')
        self.nodeTypes = sorted(nodeTypes)

        partLen = len(self.sortedSlicePaths) // self.args.num
        indx = int(self.args.indx)
        if indx == self.args.num -1:
            self.partSlices = self.sortedSlicePaths[partLen*indx :]
        else:
            self.partSlices = self.sortedSlicePaths[partLen*indx :partLen*(indx+1)]

        self.printList = []
        print(self.authors)

    def getAuthorFromPath(self, dirname):
        # in the form of "rest(api_argInde)_faddr_fname_biname", where
        # fname : L+V; biname: year_author_biname
        start = 0
        dirNameParts = basename(dirname).split("_")
        if dirNameParts[start] == 'rest':
            start = 1
        else:
            start = 2
        start += 1
        L = int(dirNameParts[start])
        start = start+1+L
        L = int(dirNameParts[start])
        author = "_".join(dirNameParts[start: start+1+L])
        return author

    def createDataset(self):
        for i, s in enumerate(self.partSlices):
            if not self.pass_slice_chk(s):
                continue
            self.createGraph4Slice(s)
            if (i+1)%(len(self.partSlices)//25)==(len(self.partSlices)//25 -1):
                print("processed {}/{}".format((i+1), len(self.partSlices)))
                # print(self.printList)
                # self.printList = []
        print("the {}th dataset created".format(i+1))

    def verify(self, slicePath):
        if '"' not in slicePath or '"' not in slicePath:
            slicePath = '"'+slicePath+'"'
        cmdChkDot = 'python3 verify_dot.py --dotDirPath '+slicePath
        proc = run(cmdChkDot, capture_output=True, shell=True)
        if proc.returncode != 0:
            print(cmdChkDot)
            print(proc.stdout.decode('utf-8'))
            print(proc.stderr.decode('utf-8'))
            if '"' not in slicePath:
                slicePath = '"'+slicePath+'"'
            delOrgDir = 'rm -r '+slicePath
            print(delOrgDir)
            run(delOrgDir, shell=True)
            try:
                self.sortedSlicePaths.remove(slicePath[1:-1])
                print("{} removed from self.sortedSlicePaths".format(slicePath))
            except:
                print("{} does not exists".format(slicePath))

    def pass_slice_chk(self, slicePath):
        self.graph_data_root = self.dataRoot
        return True

    def createGraph4Slice(self, slicePath):
        """ each subDir represents a slice, we will load their dot files about 
        AST and PDG as Networkx graphs, and then save them as pickle files, which
        are input files for our model. the files will be saved in the pathes:
        data/slice/ast_graph and data/slice/pdg_graph
        """
        # to create corresponding directories in the data directory
        # sliceDataRoot = join(self.dataRoot, basename(slicePath))
        sliceDataRoot = join(self.graph_data_root, basename(slicePath))
        if not exists(sliceDataRoot):
            os.makedirs(sliceDataRoot)
        elif len(os.listdir(sliceDataRoot)) == 2:
            return
        # get AST n PDG dot file path and load as graphs
        astPath = join(slicePath, 'ast')
        pdgPath = join(slicePath, 'pdg')
        astDotFile = join(astPath, self.astFiles[1])
        pdgDotFile = join(pdgPath, self.pdgFiles[1])
        if not exists(astDotFile):
            astDotFile = join(astPath, self.astFiles[0])
        if not exists(pdgDotFile):
            pdgDotFile = join(pdgPath, self.pdgFiles[0])
        # print("working : {}".format(slicePath))
        # load dot file as a nx graph
        g_ast = nx.drawing.nx_agraph.read_dot(astDotFile)
        g_pdg = nx.drawing.nx_agraph.read_dot(pdgDotFile)

        # to add graph corresponding author's name from file name

        author = self.getAuthorFromPath(slicePath)
        if author == None:
            return
        else:
            label = self.authors.index(author)
        g_ast.graph['label'] = g_pdg.graph['label'] = label
        g_ast.graph['sliceName'] = g_pdg.graph['sliceName'] = basename(slicePath)
        # self.printList.append(label)

        # assume four classes: 1 - control ; 2 - statement ; 3 - func call ; 4 - others
        feat = []
        for i, node in enumerate(g_pdg.node):
            # feat = np.zeros(len(PDGnodeTypes), dtype=int)
            # label form : "(<operator>.xxx, expression)"
            code = g_pdg.node.get(node).get('label').split(',')[1]
            # feat = getCodeEmbedding(code)
            tokens_ids = self.model.tokenize([code], max_length=512, mode="<encoder-only>")
            source_ids = torch.tensor(tokens_ids).to(self.device)
            _, code_embedding = self.model(source_ids)
            feat = code_embedding.squeeze().tolist()

            # check and write node label
            # depending on operator part to decide label
            first_part = g_pdg.node.get(node).get('label').split(',')[0]
            if 'operator' in first_part:
                op = first_part.split('.')[1]
                if op in self.relationalOp:
                    node_label = 1
                else:
                    node_label = 2
            else:
                if first_part in self.apiList:
                    node_label = 3
                else:
                    node_label = 4
            g_pdg.add_node(node, label=node_label, feat=np.array(feat))
        g_pdg.graph['feat_dim'] = len(feat)

        # to save pdg graph as input file of our model
        file = basename(slicePath)+"_"+str(len(feat))+"_"+str(g_pdg.number_of_nodes())+"_pdg"+self.graphpostfix
        nx.write_gpickle(g_pdg, join(sliceDataRoot, file))

        # handle nodes of an AST
        for i, node in enumerate(g_ast.node):
            feat = np.zeros(len(self.nodeTypes), dtype=int)
            cut = g_ast.node.get(node).get('label').split(',')[0]
            if 'operator' in cut:
                cut = 'operator'
            else:
                cut = cut[1:]
                if cut in self.apiList:
                    cut = 'API'
                elif cut not in self.nodeTypes:
                    cut = 'UNKNOWN'
                else:
                    pass
            feat[self.nodeTypes.index(cut)] = 1
            feat = feat.tolist()
            g_ast.add_node(node, label = 1, feat=np.array(feat))
        g_ast.graph['feat_dim']=len(self.nodeTypes)
    
        # to save ast graph as input file of our model
        file = basename(slicePath)+"_"+str(len(feat))+"_"+str(g_ast.number_of_nodes())+"_ast"+self.graphpostfix
        nx.write_gpickle(g_ast, join(sliceDataRoot, file))

class Input4attack(ModifiedDiffInput):
    """docstring for Input4attack"""
    def __init__(self, arg):
        super(Input4attack, self).__init__(arg)
        self.arg = arg
        print(self.authors)
        
    # def getAuthorFromPath(self, dirname):
    #     # "rest(api_argInde)_fname_faddr_biname"
    #     # fname: L+V
    #     # biname = round_id + challenge_id + author
    #     start = 0
    #     dirNameParts = basename(dirname)[:-len("-optx")].split("_")
    #     if dirNameParts[start] == 'rest':
    #         start = 1
    #     else:
    #         start = 2
    #     L = int(dirNameParts[start])
    #     start = start+1+L+1
    #     author = "_".join(dirNameParts[start+2:])
    #     return author
    def getAuthorFromPath(self, dirname):
        # "rest(api_argInde)_fname_faddr_biname"
        # fname: L+V
        # biname = round_id + challenge_id + 
        start = 0
        dirNameParts = basename(dirname)[:-len("-optx")].split("_")
        author = dirNameParts[-1]
        return author

    def createDataset(self):
        for i, s in enumerate(self.partSlices):
            self.createGraph4Slice(s)
            if (i+1)%(len(self.partSlices)//4)==(len(self.partSlices)//4 -1):
                print("processed {}/{}".format((i+1), len(self.partSlices)))
                # print(self.printList)
                # self.printList = []
        print("the {}th dataset created".format(i+1))

class AttackedInput(Input4attack):
    def __init__(self, arg):
        super(AttackedInput, self).__init__(arg)
        self.arg = arg
        # cmd = 'find ' + arg.attacked + ' -maxdepth 1 -type d'
        # print(cmd)
        # proc = run(cmd, capture_output=True, shell=True)
        # # 1st element is dotDirPath itself; the last one is '\n'
        # subDirs = proc.stdout.decode('utf-8').split('\n')[1:-1]
        # self.partSlices = sorted(subDirs)
        print("AttackedInput: self.partSlices length- {}".format(len(self.partSlices)))
        self.authors = sorted(['4yn', 'abhisheksaini', 'ACMonster', 'ALOHA.Brcps', 'chocimir', 'csegura', 'hoangtuanh180593', 'iPeter', 'Plypy', 'splucs'])
        self.authors = sorted(['4yn', 'hoangtuanh180593', 'iPeter'])

    # def createDataset(self):
    #     for i, s in enumerate(self.partSlices):
    #         self.createGraph4Slice(s)
    #         if (i+1)%(len(self.partSlices)//4)==(len(self.partSlices)//4 -1):
    #             print("processed {}/{}".format((i+1), len(self.partSlices)))
    #             # print(self.printList)
    #             # self.printList = []
    #     print("the {}th dataset created".format(i+1))

    # def getAuthorFromPath(self, dirname):
    #     # "rest(api_argInde)_fname_faddr_biname"
    #     # fname: L+V
    #     # biname = round_id + challenge_id + 
    #     start = 0
    #     dirNameParts = basename(dirname)[:-len("-optx")].split("_")
    #     author = dirNameParts[-1]
    #     return author

class Input4GitHub(ModifiedDiffInput):
    """docstring for Input4GitHub"""
    def __init__(self, arg):
        db_info_json = join(dirname(arg.dotpath), "db_info.json")
        with open(db_info_json, 'r') as f:
            self.db_info = json.load(f)
        super(Input4GitHub, self).__init__(arg)
        self.arg = arg
        # 'graph' directory locates the same parent as 'dot'
        # 'bls_xx' and without balance will share the same graph data
        parts = arg.dataname.split("_")
        if 'bls' in parts:
            datasetName = "_".join(parts[:-2])
        else:
            datasetName = arg.dataname
        self.graph_files_root = join(dirname(abspath(arg.dotpath)), 'graphs', datasetName)
        if not exists(self.graph_files_root):
            run("mkdir "+self.graph_files_root, shell=True)
        self.balance_num = args.bls_num

    def getNamesFromParts(self, fileNameParts):
        # slice path name contain: 'rest'/(api, argInd,), fAddr, fname, author, binary name;
        # function name, author, and binary name are in the form of 'L+V'
        # fileNameParts = splitext(basename(slice_path))[0].split('_')
        start = 0
        # fAddr start
        if fileNameParts[0] == 'rest':
            start += 1
        else:
            start += 2
        start += 1
        seperator = start
        # fname 
        l_fn_name = int(fileNameParts[start])
        start += 1
        fn_name = "_".join(fileNameParts[start:(start+l_fn_name)])

        start = start+l_fn_name
        # author name
        l_author = int(fileNameParts[start])
        start += 1
        author = "_".join(fileNameParts[start:(start+l_author)])

        start = start+l_author
        # binary name
        l_bin_name = int(fileNameParts[start])
        start += 1
        binName = "_".join(fileNameParts[start:(start+l_bin_name)])

        return fn_name, author, binName, "_".join(fileNameParts[seperator:])

    def getAuthorFromPath(self, slicePath):
        fileNameParts = basename(slicePath).split('_')
        _, author, _, _ = self.getNamesFromParts(fileNameParts)
        return author

    def createDataset(self):
        lines_2_show = 6
        if len(self.partSlices) >= lines_2_show:
            lines_in_gap = len(self.partSlices)//lines_2_show
        else:
            lines_in_gap = 1
        print("showing progress for each {} files".format(lines_in_gap))
        slices_in_db = []
        for i, s in enumerate(self.partSlices):
            pass_flag = self.pass_slice_chk(s)
            if not pass_flag:
                continue
            self.createGraph4Slice(s)
            slices_in_db.append(s)
            if (i+1)%lines_in_gap == 0:
                print("processed {}/{}".format((i+1), len(self.partSlices)))
                # print(self.printList)
                # self.printList = []
        print("the {}th dataset created".format(i+1))

        alignedSamplers = self.adjust_authors_2_same_samples(slices_in_db)
        print("alignedSamplers len: {}".format(len(alignedSamplers)))
        # to keep balance and mv to the target
        # to calculate exact cp num as
        fp = open("kp_bls_"+self.args.indx+".log", 'w')
        for slice_path in slices_in_db:
            _, _, _, fn_author_bin = self.getNamesFromParts(basename(slice_path).split("_"))
            cp_num = alignedSamplers.count(fn_author_bin)
            for idx in range(cp_num):
                new_slice_path = join(self.dataRoot, basename(slice_path)+"-cp"+str(idx))
                try:
                    src = join(self.graph_files_root, basename(slice_path))
                    shutil.copytree(src, new_slice_path)
                    fp.write("{} copied to {}\n".format(src, new_slice_path))
                except FileExistsError:
                    # raise e
                    fp.write("Destination dir {} already exists\n".format(new_slice_path))
                # fp.write("\t{}\n".format(cmd))

        fp.close()
        print("keep balance done!")

    def pass_slice_chk(self, slicePath):
        sliceNameParts = basename(slicePath).split("_")
        _, _, _, normFn = self.getNamesFromParts(sliceNameParts)

        if normFn in self.db_info["all_fns"]:
            self.graph_data_root = self.graph_files_root
            return True
        else:
            return False
        
    def getAuthorsSet(self):
        authors = set()
        # print("getting github authors, len sub dir is {}".format(len(self.sortedSlicePaths)))
        for subdir in self.sortedSlicePaths:
            sliceNameParts = basename(subdir).split("_")
            _, author, _, normFn = self.getNamesFromParts(sliceNameParts)
            # print(author, slice_name)
            # x=input()
            if normFn in self.db_info["all_fns"]:
                authors.add(author)

        self.authors = sorted(list(authors))
        print("there are {} authors. They are: \n{}".format(len(self.authors), self.authors))

    def adjust_authors_2_same_samples(self, allSlicePaths):
        all_fns = []
        # to create relationship between author and fn from all slice paths
        author_fn_dict = dict()
        for slicePath in allSlicePaths:
            _, author, _, normFn = self.getNamesFromParts(basename(slicePath).split("_"))
            Util.add_element_as_list_2_key_in_dict(normFn, author, author_fn_dict)

        for author in author_fn_dict.keys():
            author_fn_dict[author] = list(set(author_fn_dict[author]))

        # to make all authors have the same sample fns
        self.sample(author_fn_dict)

        for author in author_fn_dict.keys():
            all_fns += author_fn_dict[author]

        return all_fns

    def sample(self, author_fn_dict):
        if self.balance_num == 0:
            return

        if self.args.under_sample:
            self.under_sample(author_fn_dict)

        if self.args.up_sampler:
            self.up_sampler(author_fn_dict)
                          
    def under_sample(self, author_fn_dict):
        for author in author_fn_dict.keys():
            if len(author_fn_dict[author]) > self.balance_num:
                balanced_smpl = random.sample(author_fn_dict[author], self.balance_num)
                author_fn_dict[author] = balanced_smpl

    def up_sampler(self, author_fn_dict):
        for author in author_fn_dict.keys():
            if len(author_fn_dict[author]) <= self.balance_num:
                original_len = len(author_fn_dict[author])
                repeated_num = self.balance_num // original_len
                tail_num = self.balance_num - (repeated_num * original_len)
                assert tail_num >= 0, (original_len, repeated_num, tail_num)
                if tail_num > 0:
                    tailer = random.sample(author_fn_dict[author], tail_num)
                else:
                    tailer = []
                balanced_smpl = author_fn_dict[author] * repeated_num +tailer
                author_fn_dict[author] = balanced_smpl

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', "--datadir", type=str, help="dir to store data files", required=True)
    parser.add_argument('-n', "--dataname", type=str, help="name for data set", required=True)
    parser.add_argument('-dot', "--dotpath", type=str, help="the path of 'dot dir contains ASTs", default=None)
    parser.add_argument('-sf',help="if only create specific file or not. for Class DiffInput", default=False, action='store_true')
    parser.add_argument('-f', "--file",help="specific file. For Class DiffInput", default=None, type=str)
    parser.add_argument('-N', "--num", type=int, help="the number of threads", default=1)
    parser.add_argument('-i', "--indx", type=str, help="the index of this part", default=0)
    parser.add_argument('-bn',  type=int, help="smpale num that keeps the db balance ", default=0, dest="bls_num")
    parser.add_argument('-up_sample', help="using up sampling tech", action='store_true', dest="up_sampler", default=False)
    parser.add_argument('-under_sample', help="using under sampling tech", action='store_true', dest="under_sample", default=False)
    args = parser.parse_args()
    # diff_inputs = DiffInput(args)
    diff_inputs = ModifiedDiffInput(args)
    # if args.attacked:
    #     diff_inputs = AttackedInput(args)
    # else:
    #     diff_inputs = Input4attack(args)
    # diff_inputs = Input4GitHub(args)

    if args.sf != True:
        diff_inputs.createDataset()
    else:
        if args.file == None:
            print("missing specific file name")
        else:
            diff_inputs.createSpecificFile(args.file)
