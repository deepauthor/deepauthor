import matplotlib
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import networkx as nx
import numpy as np
import sklearn.metrics as metrics
from sklearn.utils.class_weight import compute_class_weight
import torch
import torch.nn as nn
from torch.autograd import Variable
import tensorboardX
from tensorboardX import SummaryWriter

import argparse
import os
import pickle
import random
import shutil
import time

import cross_val
import encoders
# import gen.feat as featgen
# import gen.data as datagen    
from graph_sampler import GraphSampler
import load_data
import util

import warnings
warnings.filterwarnings("ignore")
import mkArgs
from torch.utils.data import DataLoader, Dataset
from siamese_arch import SiameseArch, SiameseDataset_fn, SiameseArch_fn, Focal_loss, SiameseArch_fn_weighted
# from memory_profiler import profile
from os.path import basename, isdir, join, dirname, exists
from subprocess import run
import cProfile

import resource

# Increase the shared memory limit
soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_AS)
new_soft_limit = 2 * soft_limit  # Adjust the multiplier as per your requirement
resource.setrlimit(resource.RLIMIT_AS, (new_soft_limit, hard_limit))

def prepare_val_data(fn_graphs, slices_graphs, args, val_idx, max_nodes=0):
    graphs = fn_graphs["training"]
    # generate slices_pairs
    # choose any two fn as a fn pair
    fn_index_pairs = set()
                                                               
    comb_factor = 2
    while (len(graphs)*comb_factor)>len(fn_index_pairs):
        # make combination from slices
        i, j = random.sample(range(len(graphs)), 2)
        if (i, j) in fn_index_pairs or (j, i) in fn_index_pairs:
            continue
        else:
            fn_index_pairs.add((i,j))

    # check and cover all fns
    chosen_fn = []
    [chosen_fn.extend(fn_pair) for fn_pair in fn_index_pairs]
    chosen_fn = list(set(chosen_fn))
    missed_fn = [idx for idx in list(range(len(graphs))) if idx not in chosen_fn]
    while len(missed_fn) > 0:
        i, j = random.sample(range(len(graphs)), 2)
        if i != missed_fn[0]:
            fn_index_pairs.add((missed_fn[0], i))
        if j != missed_fn[0]:
            fn_index_pairs.add((missed_fn[0], j))
        # update missed_fn
        if i in missed_fn:
            missed_fn.remove(i)
        if j in missed_fn:
            missed_fn.remove(j)

        missed_fn = missed_fn[1:]

    # prediction also cover all fns
    graphs = []
    for fn_pair in list(fn_index_pairs):
        x, y = fn_pair
        graphs.append(fn_pair)
        graphs.append((y, x))
    train_graphs = graphs

    val_graphs = fn_graphs["test"]
    val_graphs = [(val_graphs.index(e), 0) for e in val_graphs]

    print('Num training graphs: ', len(train_graphs), 
          '; Num validation graphs: ', len(val_graphs))

    print('Number of graphs: ', len(train_graphs)+len(val_graphs))

    # minibatch
    train_dataset = SiameseDataset_fn(train_graphs, fn_graphs["training"], slices_graphs, max_num_nodes=max_nodes)
    # , features=args.feature_type)
    train_dataset_loader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=args.batch_size, 
            shuffle=True,
            pin_memory=True,
            num_workers=args.num_workers)

    # val_dataset = SiameseDataset_fn(val_graphs, fn_graphs, slices_graphs, normalize=False, max_num_nodes=max_nodes,
    #         features=args.feature_type)
    val_dataset = SiameseDataset_fn(val_graphs, fn_graphs["test"], slices_graphs, max_num_nodes=max_nodes)
    val_dataset_loader = torch.utils.data.DataLoader(
            val_dataset, 
            batch_size=args.batch_size, 
            shuffle=False,
            pin_memory=True,
            num_workers=args.num_workers)

    return train_dataset_loader, val_dataset_loader, \
            train_dataset.max_num_nodes, train_dataset.feat_dim, train_dataset.assign_feat_dim
        
def get_data_from_fn(fn, device):
    ret = []
    data = fn
    for s in data['slices']:
        adj_ast = Variable(s['ast']['adj'].float(), requires_grad=False).to(device)
        h0_ast = Variable(s['ast']['feats'].float()).to(device)
        batch_num_nodes_ast = s['ast']['num_nodes'].int().numpy()
        assign_input_ast = Variable(s['ast']['assign_feats'].float(), requires_grad=False).to(device)
        adj_pdg = Variable(s['pdg']['adj'].float(), requires_grad=False).to(device) # by  cuda()
        h0_pdg = Variable(s['pdg']['feats'].float()).to(device) # by  cuda()
        # labels.append(s['pdg']['label'].long().numpy())
        batch_num_nodes_pdg = s['pdg']['num_nodes'].int().numpy()
        assign_input_pdg = Variable(s['pdg']['assign_feats'].float(), requires_grad=False).to(device) # by  cuda()
        ret.append((h0_ast, adj_ast, batch_num_nodes_ast, assign_input_ast, 
                    h0_pdg, adj_pdg, batch_num_nodes_pdg, assign_input_pdg))

    return ret

def model_evaluate(dataset, model, args, name='Validation', max_num_examples=None):
    # # load our model first if exists
    # if exists(args.save_path):
    #     model_info = torch.load(args.save_path)
    #     model.load_state_dict(model_info['model_state_dict'])
    model.eval()

    labels = []
    preds = []
    if args.cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
    # combination embedding by 
    # for batch_idx, data in enumerate(dataset):
    batch_idx = 0
    for data in dataset:
        fn1, fn2, _, author_label = data

        fn1_data = get_data_from_fn(fn1, device)
        fn2_data = get_data_from_fn(fn2, device)
        labels.append(author_label.long().numpy())

                                  
        _, ypred = model(fn1_data, fn2_data)
        _, indices = torch.max(ypred, 1)
        # numpy doesn't support GPU
        preds.append(indices.cpu().data.numpy())

        if max_num_examples is not None:
            if (batch_idx+1)*args.batch_size > max_num_examples:
                break
        batch_idx += 1

    labels = np.hstack(labels)
    preds = np.hstack(preds)
    
    result = {'prec': metrics.precision_score(labels, preds, average='macro'),
              'recall': metrics.recall_score(labels, preds, average='macro'),
              'acc': metrics.accuracy_score(labels, preds),
              'F1': metrics.f1_score(labels, preds, average="micro")}
    print(name, " accuracy:", result['acc'])
    return result

# @profile(stream=fp)
def model_train(train_dataset, model, args, same_feat=True, val_dataset=None, test_dataset=None, writer=None, mask_nodes = True):
    optimizer = torch.optim.Adam(filter(lambda p : p.requires_grad, model.parameters()), lr=0.001)
    iter = 0
    best_val_result = {
            'epoch': 0,
            'loss': 0,
            'acc': 0}
    test_result = {
            'epoch': 0,
            'loss': 0,
            'acc': 0}
    train_accs = []
    train_epochs = []
    best_val_accs = []
    best_val_epochs = []
    test_accs = []
    test_epochs = []
    val_accs = []

    device = torch.device("cuda")

    for epoch in range(args.num_epochs):
        # if epoch <= last_epoch:
        #     continue
        total_time = 0
        avg_loss = 0.0
        model.train()
        print('Epoch: ', epoch)
        # enumerate has bugs with cuda so change it by 
        # for batch_idx, data in enumerate(dataset):
        batch_idx = 0

        # begin_time = time.time()
        for data in train_dataset:
            model.zero_grad()
            fn1, fn2, similarity_label, author_label = data

            fn1_data = get_data_from_fn(fn1, device)

            fn2_data = get_data_from_fn(fn2, device)
                                                                                                 
            similarity_label = Variable(similarity_label.long()).to(device)
            author_label = Variable(author_label.long()).to(device)
            similarity_pre, author_pre = model(fn1_data, fn2_data)
                                                                       
            loss1 = model.loss1(similarity_pre, similarity_label.float())
            loss2 = model.loss2(author_pre, author_label)
            loss = loss1 + loss2
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            iter += 1
            avg_loss += loss
            #if iter % 20 == 0:
            #    print('Iter: ', iter, ', loss: ', loss.data[0])
            batch_idx += 1
            # save model for each 25 epochs
            if epoch//25 == 0 and batch_idx == len(train_dataset) // 2 and exists(args.save_path):
                model_info = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                }
                torch.save(model_info, args.save_path)


        # avg_loss /= batch_idx + 1
        avg_loss /= batch_idx
        # elapsed = time.time() - begin_time
        # total_time += elapsed

        # print('Avg loss: ', avg_loss, '; epoch time: ', elapsed)
        print('Avg loss: ', avg_loss)
        # result = evaluate(dataset, model, args, name='Train', max_num_examples=100)
        result = model_evaluate(train_dataset, model, args, name='train', max_num_examples=100)
        train_accs.append(result['acc'])
        train_epochs.append(epoch)
        if val_dataset is not None:
            val_result = model_evaluate(val_dataset, model, args, name='validation')
            val_accs.append(val_result['acc'])
        if val_result['acc'] > best_val_result['acc'] - 1e-7:
            best_val_result['acc'] = val_result['acc']
            best_val_result['epoch'] = epoch
            best_val_result['loss'] = avg_loss
        if test_dataset is not None:
            test_result = model_evaluate(test_dataset, model, args, name='test')
            test_result['epoch'] = epoch

        print('best val result: ', best_val_result)
        best_val_epochs.append(best_val_result['epoch'])
        best_val_accs.append(best_val_result['acc'])
        if test_dataset is not None:
            print('test result: ', test_result)
            test_epochs.append(test_result['epoch'])
            test_accs.append(test_result['acc'])

    # save our model after training
    model_info = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(model_info, args.save_path)

    return model, val_accs

def get_test_cases(slice_dirs):
    print("choosing parts of slices")
    tmp_dict = dict()
    for e in slice_dirs:
        slice_name = basename(e)
        _, author_name = get_fname_author_from_string(slice_name)

        if author_name not in tmp_dict.keys():
            tmp_dict[author_name] = [e]
        else:
            tmp_dict[author_name].append(e)
    tmp = []
    if 10 // len(tmp_dict.keys()) >= 1:
        num = 2 + (10 // len(tmp_dict.keys()))
    else:
        num = 2
    # print("case cp num is {}".format(num))
    for author in tmp_dict.keys():
        # rnd = random.randint(0, len(tmp_dict[author])-1)
        rnd = random.sample(tmp_dict[author], num)
        # print("{} got {}".format(author, [basename(e) for e in rnd]))
        tmp += rnd

    return tmp

def get_fname_author_from_string(string):
    # slice file name contain: 'rest'/(api, argInd,), fAddr, fname, author, binary name;
    # function name, author, and binary name are in the form of 'L+V'
    fileNameParts = string.split('_')
    start = 0
    # fAddr start
    if fileNameParts[0] == 'rest':
        start += 1
    else:
        start += 2
    # fname start
    start += 1
    l_fn_name = int(fileNameParts[start])
    start += 1
    fn_name = "_".join(fileNameParts[start:])

    start = start+l_fn_name
    l_author = int(fileNameParts[start])
    start += 1
    author = "_".join(fileNameParts[start:(start+l_author)])

    return fn_name, author

def get_aligned_fn_graphs_from_slice_graphs(slices_graphs):
    fns_slice_dict = dict()
    for e in slices_graphs:
        slice_name = e[0].graph['sliceName']
        func_name, _ = get_fname_author_from_string(slice_name)

        if func_name not in fns_slice_dict.keys():
            fns_slice_dict[func_name] = [slices_graphs.index(e)]
        else:
            fns_slice_dict[func_name].append(slices_graphs.index(e))
    max_slice_num = 0
    for fname in fns_slice_dict.keys():
        if len(fns_slice_dict[fname]) > max_slice_num:
            max_slice_num = len(fns_slice_dict[fname])

    print(max_slice_num )

    # to align to the max slices 
    for fname in fns_slice_dict.keys():
        fns_slice_dict[fname] += [-1]*(max_slice_num - len(fns_slice_dict[fname]))

    fn_list = list(fns_slice_dict.values())

    return fn_list

def getWeight(train_dataset):
    author_labels = []
    simil_labels = []
    for data in train_dataset:
        _, _, similarity, author_label = data
        author_labels.append(author_label.long().numpy())
        simil_labels.append(similarity.long().numpy())

    author_labels = np.hstack(author_labels)
    print(author_label.shape)
    simil_labels = np.hstack(simil_labels)
    author_weight = compute_class_weight(class_weight="balanced", classes=np.unique(author_labels), y=author_labels)
    simil_labels_weight = compute_class_weight(class_weight="balanced", classes=np.unique(simil_labels), y=simil_labels)
    # following the formular of compute_class_weight, to calculate the number
    # of a weight of positive examples for 'BCEWithLogitsLoss' for binary classification
    simil_weight = simil_labels_weight[0]/simil_labels_weight[1] 
    return author_weight, simil_weight

def benchmark_task_val(args, writer=None, feat='node-feat'):
    all_vals = []
    # change input files to individual sub-directories by 
    target_path = os.path.join(args.datadir, args.bmname)
    slice_dirs = [join(target_path, d) for d in os.listdir(target_path) if isdir(join(target_path, d))]

    # for testing
    if args.test:
        slice_dirs = get_test_cases(slice_dirs)
        print("end test choosing")
        # print("len slices are {}".format(len(slice_dirs)))

    slices_graphs = load_data.read_dot_files(slice_dirs, args.max_nodes)
    # slices_graphs = load_data.read_dot_files_github(slice_dirs_dict, args.max_nodes)

    fn_list = get_aligned_fn_graphs_from_slice_graphs(slices_graphs)
    random.shuffle(fn_list)

    device = torch.device("cuda")
    k = 10
    if k == 1:
        val_size = len(fn_list)  // k
    else:
        val_size = len(fn_list)  // 10
    for i in range(k):
        print("k: {} round".format(i))
        begin_time = time.time()
        train_test_dict = {"training":fn_list[:i*val_size]+fn_list[(i+1)*val_size:], "test":fn_list[i*val_size: (i+1)*val_size]}
        # print(train_test_dict)
        train_dataset, val_dataset, max_num_nodes, input_dim, assign_input_dim = \
            prepare_val_data(train_test_dict, slices_graphs, args, i, max_nodes=args.max_nodes)

        author_weight, simil_weight = getWeight(train_dataset)
        author_weight = torch.from_numpy(author_weight).float().to(device)
        simil_weight = torch.tensor([simil_weight]).float().to(device)
        weight = {"classifier":author_weight, "siamese":simil_weight}
        # print(weight)
        print("data prepare finished")

        if args.method == 'soft-assign':
            print('Method: soft-assign')
            # model = encoders.SoftPoolingGcnEncoder(
            # model = encoders.SoftPoolingGcnEncoder_4_ast_n_pdg(
            model = SiameseArch_fn_weighted( max_num_nodes,
                    input_dim, args.hidden_dim, args.output_dim, args.num_classes, args.num_gc_layers,
                    args.hidden_dim, assign_input_dim, weight, assign_ratio=args.assign_ratio, num_pooling=args.num_pool,
                    bn=args.bn, dropout=args.dropout, linkpred=args.linkpred, args=args).to(device) # by  cuda()
            print("model building finished")
        _, val_accs = model_train(train_dataset, model, args, val_dataset=val_dataset, test_dataset=None)
            # writer=writer)
        all_vals.append(np.array(val_accs))
        elapsed = time.time() - begin_time
        print("{} round spend {} s".format(i, elapsed))
    print(all_vals)
    all_vals = np.vstack(all_vals)
    all_vals = np.mean(all_vals, axis=0)
    print(all_vals)
    print(np.max(all_vals))
    print(np.argmax(all_vals))


def main():
    prog_args = mkArgs.arg_parse()


    if prog_args.cuda != None:
        os.environ['CUDA_VISIBLE_DEVICES'] = prog_args.cuda # by yihu
    # os.environ['CUDA_VISIBLE_DEVICES'] = ""
    print('CUDA', prog_args.cuda)

    if prog_args.bmname is not None:
        # benchmark_task_val(prog_args, writer=writer)
        if not exists(dirname(prog_args.save_path)):
            run('mkdir -p '+dirname(prog_args.save_path), shell=True)
        benchmark_task_val(prog_args)


if __name__ == "__main__":
    # profiler = cProfile.Profile()
    # profiler.enable()
    main()
    # mainprofiler.disable()
    # profiler.print_stats()
