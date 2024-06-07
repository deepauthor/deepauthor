import encoders
import torch
import torch.nn as nn
import torch.nn.functional as F
import encoders
from torch.nn import init
from torch.utils.data import DataLoader, Dataset
import networkx as nx
import numpy as np
import torch
import torch.utils.data
from os.path import basename, splitext, join
import util

class Focal_loss(nn.Module):
    def __init__(self, alpha=None, gamma=2):
        super(Focal_loss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.nll_loss = nn.NLLLoss(weight=alpha, reduction='none')

    def forward(self, inputs, targets):
        x = inputs
        y = targets
        # compute weighted cross entropy term: -alpha * log(pt)
        # (alpha is already part of self.nll_loss)
        log_p = F.log_softmax(x, dim=-1)
        ce = self.nll_loss(log_p, y)

        # get true class column from each row
        all_rows = torch.arange(len(x))
        log_pt = log_p[all_rows, y]

        # compute focal term: (1 - pt)^gamma
        pt = log_pt.exp()
        focal_term = (1 - pt)**self.gamma

        # the full loss: -alpha * ((1 - pt)^gamma) * log(pt)
        loss = focal_term*ce

        loss = loss.mean()

        return loss

class SiameseArch(encoders.SoftPoolingGcnEncoder_my_module_base):
    def __init__(self, max_num_nodes, input_dim_dict, hidden_dim, embedding_dim, label_dim, num_layers,
            assign_hidden_dim, assign_input_dim_dict, assign_ratio=0.25, assign_num_layers=-1, num_pooling=1,
            pred_hidden_dims=[50], concat=True, bn=True, dropout=0.0, linkpred=True, args=None):

        key = 'ast'
        input_dim = input_dim_dict[key]
        assign_input_dim = assign_input_dim_dict[key]
        super(SiameseArch, self).__init__(
            max_num_nodes, input_dim, hidden_dim, embedding_dim, label_dim, num_layers,
            assign_hidden_dim, assign_ratio=assign_ratio, num_pooling=num_pooling, bn=bn, dropout=dropout,
            linkpred=linkpred, assign_input_dim=assign_input_dim, args=args)
        self.ast_module = encoders.SoftPoolingGcnEncoder_my_module_base(
            max_num_nodes, input_dim, hidden_dim, embedding_dim, label_dim, num_layers,
            assign_hidden_dim, assign_ratio=assign_ratio, num_pooling=num_pooling, bn=bn, dropout=dropout,
            linkpred=linkpred, assign_input_dim=assign_input_dim, args=args)
        self.pred_input_dim_ast = self.pred_input_dim*(self.num_pooling+1)

        key = 'pdg'
        input_dim = input_dim_dict[key]
        assign_input_dim = assign_input_dim_dict[key]
        self.pdg_module = encoders.SoftPoolingGcnEncoder_my_module_base(
            max_num_nodes, input_dim, hidden_dim, embedding_dim, label_dim, num_layers,
            assign_hidden_dim, assign_ratio=assign_ratio, num_pooling=num_pooling, bn=bn, dropout=dropout,
            linkpred=linkpred, assign_input_dim=assign_input_dim, args=args)
        self.pred_input_dim_pdg = self.pred_input_dim*(self.num_pooling+1)

        # trying concatanation of pdg and ast
        # self.pred_model = self.build_pred_layers(self.pred_input_dim_ast+self.pred_input_dim_pdg, pred_hidden_dims, 
        # trying sum/average/weighted sum of pdg and ast, so the same shape with dim of ast or pdg
        pre_input_dim = self.pred_input_dim_ast
        self.pred_model = self.build_pred_layers(pre_input_dim, pred_hidden_dims, label_dim, num_aggs=self.num_aggs)

        self.criterion = nn.BCEWithLogitsLoss()
        for m in self.modules():
            if isinstance(m, encoders.GraphConv):
                m.weight.data = init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    m.bias.data = init.constant(m.bias.data, 0.0)

        
    def forward_once(self, x_ast, adj_ast, batch_num_nodes_ast, x_pdg, adj_pdg, batch_num_nodes_pdg, **kwargs):
        if 'assign_x_ast' in kwargs:
            x_a = kwargs['assign_x_ast']
        else:
            x_a = x_ast
        # print("shapes: x_ast {}\t adj_ast {} ".format(x_ast.size(), adj_ast.size()))
        ast_output = self.ast_module( x_ast, adj_ast, batch_num_nodes_ast, assign_x=x_a)
        # print("ast_output shape {}".format(ast_output.size()))
        # self.assign_tensor_ast = self.ast_module.assign_tensor

        if 'assign_x_pdg' in kwargs:
            x_a = kwargs['assign_x_pdg']
        else:
            x_a = x_pdg
        # print("shapes: x_pdg {}\t adj_pdg {} ".format(x_pdg.size(), adj_pdg.size()))
        pdg_output = self.pdg_module( x_pdg, adj_pdg, batch_num_nodes_pdg, assign_x=x_a)
        # print("pdg_output shape {}".format(pdg_output.size()))
        output = torch.add(ast_output, pdg_output)
        return output
        
    def forward(self, s1, s2):
        x_ast1, adj_ast1, batch_num_nodes_ast1, x_pdg1, adj_pdg1, batch_num_nodes_pdg1, kwargs1 = s1
        x_ast2, adj_ast2, batch_num_nodes_ast2, x_pdg2, adj_pdg2, batch_num_nodes_pdg2, kwargs2 = s2
        coding_style_embedding1 = self.forward_once(x_ast1, adj_ast1, batch_num_nodes_ast1, x_pdg1, adj_pdg1, batch_num_nodes_pdg1, **kwargs1)
        coding_style_embedding2 = self.forward_once(x_ast2, adj_ast2, batch_num_nodes_ast2, x_pdg2, adj_pdg2, batch_num_nodes_pdg2, **kwargs2)
        similarity_pre = F.cosine_similarity(coding_style_embedding1, coding_style_embedding2)
        # pre_input = torch.add(ast_output, pdg_output)
        author_pre = self.pred_model(coding_style_embedding1)
        return similarity_pre, author_pre

    def loss1(self, preds, labels):
        # for similarity prediction loss value
        loss = self.criterion(preds, labels)
        return loss

    def loss2(self, preds, labels):
        # for author prediction loss value
        loss = super(SiameseArch, self).loss(preds, labels)
        return loss

class SiameseArch_fn(encoders.SoftPoolingGcnEncoder_my_module_base):
    """the model for computing vector of a function and predict its author by using Siamese Arch"""
    def __init__(self, max_num_nodes, input_dim_dict, hidden_dim, embedding_dim, label_dim, num_layers,
            assign_hidden_dim, assign_input_dim_dict, assign_ratio=0.25, assign_num_layers=-1, num_pooling=1,
            pred_hidden_dims=[50], concat=True, bn=True, dropout=0.0, linkpred=True, args=None):
        key = 'ast'
        input_dim = input_dim_dict[key]
        assign_input_dim = assign_input_dim_dict[key]
        super(SiameseArch_fn, self).__init__(max_num_nodes, input_dim, hidden_dim, embedding_dim, label_dim, num_layers,
            assign_hidden_dim, assign_ratio=assign_ratio, num_pooling=num_pooling, bn=bn, dropout=dropout,
            linkpred=linkpred, assign_input_dim=assign_input_dim, args=args)

        self.ast_module = encoders.SoftPoolingGcnEncoder_my_module_base(
            max_num_nodes, input_dim, hidden_dim, embedding_dim, label_dim, num_layers,
            assign_hidden_dim, assign_ratio=assign_ratio, num_pooling=num_pooling, bn=bn, dropout=dropout,
            linkpred=linkpred, assign_input_dim=assign_input_dim, args=args)
        self.pred_input_dim_ast = self.pred_input_dim*(self.num_pooling+1)

        key = 'pdg'
        input_dim = input_dim_dict[key]
        assign_input_dim = assign_input_dim_dict[key]
        self.pdg_module = encoders.SoftPoolingGcnEncoder_my_module_base(
            max_num_nodes, input_dim, hidden_dim, embedding_dim, label_dim, num_layers,
            assign_hidden_dim, assign_ratio=assign_ratio, num_pooling=num_pooling, bn=bn, dropout=dropout,
            linkpred=linkpred, assign_input_dim=assign_input_dim, args=args)
        
        pre_input_dim = self.pred_input_dim_ast
        self.pred_model = self.build_pred_layers(pre_input_dim, pred_hidden_dims, label_dim, num_aggs=self.num_aggs)

        self.criterion = nn.BCEWithLogitsLoss()
        # self.criterion = nn.BCEWithLogitsLoss(pos_weight=weight["siamese"])

        # self.focal_loss = Focal_loss(weight["classifier"])

    def forward(self, f1, f2):
        fn_feat = []
        for fn in [f1, f2]:
            slice_feats = []
            for s in fn:
                x_ast, adj_ast, batch_num_nodes_ast, assign_ast, x_pdg, adj_pdg, batch_num_nodes_pdg, assign_pdg = s
                                          
                                              
                       
                                                      
                ast_output = self.ast_module(x_ast, adj_ast, batch_num_nodes_ast, assign_x=assign_ast)
                pdg_output = self.pdg_module(x_pdg, adj_pdg, batch_num_nodes_pdg, assign_x=assign_pdg)
                output = torch.add(ast_output, pdg_output)
                                               
                # batch_num_nodes_ast :(batch-size, ); '0' means the slice is a padded slice.
                for i in range(len(batch_num_nodes_ast)):
                    if batch_num_nodes_ast[i] == 0:
                                                                               
                        output[i] = torch.zeros(self.pred_input_dim_ast)
                                                                                                         
                                                                             
                                                   
                slice_feats.append(output)

            fn_feat.append( torch.stack(slice_feats).sum(dim=0) )

        similarity_pre = F.cosine_similarity(fn_feat[0], fn_feat[1])

        author_pre = self.pred_model(fn_feat[0])

        return similarity_pre, author_pre

    def loss1(self, preds, labels):
        # for similarity prediction loss value
        loss = self.criterion(preds, labels)
        return loss

    def loss2(self, preds, labels):
        # for author prediction loss value
        # loss = super(SiameseArch_fn, self).loss(preds, labels)
        loss = F.cross_entropy(preds, labels) 
        # loss = self.focal_loss(preds, labels)
        return loss

class SiameseArch_fn_weighted(SiameseArch_fn):
    def __init__(self, max_num_nodes, input_dim_dict, hidden_dim, embedding_dim, label_dim, num_layers,
            assign_hidden_dim, assign_input_dim_dict, weight, assign_ratio=0.25, assign_num_layers=-1, num_pooling=1,
            pred_hidden_dims=[50], concat=True, bn=True, dropout=0.0, linkpred=True, args=None):
        super(SiameseArch_fn_weighted, self).__init__(max_num_nodes, input_dim_dict, hidden_dim, embedding_dim, label_dim, num_layers,
            assign_hidden_dim, assign_input_dim_dict, assign_ratio=assign_ratio, num_pooling=num_pooling, bn=bn, dropout=dropout,
            linkpred=linkpred, args=args)


        self.criterion = nn.BCEWithLogitsLoss(pos_weight=weight["siamese"])

        self.focal_loss = Focal_loss(weight["classifier"])

    def loss2(self, preds, labels):
        # for author prediction loss value
        # loss = super(SiameseArch_fn, self).loss(preds, labels)
        # loss = F.cross_entropy(preds, labels) 
        loss = self.focal_loss(preds, labels)
        return loss

class DeepAuthor_without_siamese_fn(encoders.SoftPoolingGcnEncoder_my_module_base):
    """docstring for DeepAuthor_without_siamese_fn"""
    def __init__(self, max_num_nodes, input_dim_dict, hidden_dim, embedding_dim, label_dim, num_layers,
            assign_hidden_dim, assign_input_dim_dict, weight, assign_ratio=0.25, assign_num_layers=-1, num_pooling=1,
            pred_hidden_dims=[50], concat=True, bn=True, dropout=0.0, linkpred=True, args=None):
        # to test coding style in only AST or only PDG
        if input_dim_dict['ast'] != None:
            key = 'ast'
        else:
            key = 'pdg'
        self.key = key
        input_dim = input_dim_dict[key]
        assign_input_dim = assign_input_dim_dict[key]
        super(SiameseArch_fn, self).__init__(max_num_nodes, input_dim, hidden_dim, embedding_dim, label_dim, num_layers,
            assign_hidden_dim, assign_ratio=assign_ratio, num_pooling=num_pooling, bn=bn, dropout=dropout,
            linkpred=linkpred, assign_input_dim=assign_input_dim, args=args)

        self.ast_or_pdg_module = encoders.SoftPoolingGcnEncoder_my_module_base(
            max_num_nodes, input_dim, hidden_dim, embedding_dim, label_dim, num_layers,
            assign_hidden_dim, assign_ratio=assign_ratio, num_pooling=num_pooling, bn=bn, dropout=dropout,
            linkpred=linkpred, assign_input_dim=assign_input_dim, args=args)
        self.pred_input_dim_ast = self.pred_input_dim*(self.num_pooling+1)

        # key = 'pdg'
        # input_dim = input_dim_dict[key]
        # assign_input_dim = assign_input_dim_dict[key]
        # self.pdg_module = encoders.SoftPoolingGcnEncoder_my_module_base(
        #     max_num_nodes, input_dim, hidden_dim, embedding_dim, label_dim, num_layers,
        #     assign_hidden_dim, assign_ratio=assign_ratio, num_pooling=num_pooling, bn=bn, dropout=dropout,
        #     linkpred=linkpred, assign_input_dim=assign_input_dim, args=args)
        
        pre_input_dim = self.pred_input_dim_ast
        self.pred_model = self.build_pred_layers(pre_input_dim, pred_hidden_dims, label_dim, num_aggs=self.num_aggs)

        self.criterion = nn.BCEWithLogitsLoss(pos_weight=weight["siamese"])

        self.focal_loss = Focal_loss(weight["classifier"])

    def forward(self, f1, _):
        # fn_feat = []
        # for fn in [f1, f2]:
        slice_feats = []
        for s in fn:
            x_ast, adj_ast, batch_num_nodes_ast, assign_ast, x_pdg, adj_pdg, batch_num_nodes_pdg, assign_pdg = s
            # ast_output = self.ast_module(x_ast, adj_ast, batch_num_nodes_ast, assign_x=assign_ast)
            # pdg_output = self.pdg_module(x_pdg, adj_pdg, batch_num_nodes_pdg, assign_x=assign_pdg)
            # output = torch.add(ast_output, pdg_output)
            if self.key == 'ast':
                output = self.ast_module(x_ast, adj_ast, batch_num_nodes_ast, assign_x=assign_ast)
            else:
                output = self.pdg_module(x_pdg, adj_pdg, batch_num_nodes_pdg, assign_x=assign_pdg)
                                           
            # batch_num_nodes_ast :(batch-size, ); '0' means the slice is a padded slice.
            for i in range(len(batch_num_nodes_ast)):
                if batch_num_nodes_ast[i] == 0:
                    output[i] = torch.zeros(self.pred_input_dim_ast)
                                               
            slice_feats.append(output)

        # fn_feat.append( torch.stack(slice_feats).sum(dim=0) )
        fn_feat = torch.stack(slice_feats).sum(dim=0)

        # similarity_pre = F.cosine_similarity(fn_feat[0], fn_feat[1])

        # author_pre = self.pred_model(fn_feat[0])
        author_pre = self.pred_model(fn_feat)

        return similarity_pre, author_pre

    def loss1(self, preds, labels):
        # for similarity prediction loss value
        # loss = self.criterion(preds, labels)
        # return loss
        return Variable(torch.Tensor([0]).type_as(class_loss.data))

    def loss2(self, preds, labels):
        # for author prediction loss value
        # loss = super(SiameseArch_fn, self).loss(preds, labels)
        # loss = F.cross_entropy(preds, labels) 
        loss = self.focal_loss(preds, labels)
        return loss
        
class SiameseArch_ast_pdg_fn_weighted(encoders.SoftPoolingGcnEncoder_my_module_base):
    def __init__(self, max_num_nodes, input_dim_dict, hidden_dim, embedding_dim, label_dim, num_layers,
            assign_hidden_dim, assign_input_dim_dict, weight, assign_ratio=0.25, assign_num_layers=-1, num_pooling=1,
            pred_hidden_dims=[50], concat=True, bn=True, dropout=0.0, linkpred=True, args=None):
        key = 'ast'
        input_dim = input_dim_dict[key]
        assign_input_dim = assign_input_dim_dict[key]
        super(SiameseArch_ast_pdg_fn_weighted, self).__init__(max_num_nodes, input_dim, hidden_dim, embedding_dim, label_dim, num_layers,
            assign_hidden_dim, assign_ratio=assign_ratio, num_pooling=num_pooling, bn=bn, dropout=dropout,
            linkpred=linkpred, assign_input_dim=assign_input_dim, args=args)

        self.ast_module = encoders.SoftPoolingGcnEncoder_my_module_base(
            max_num_nodes, input_dim, hidden_dim, embedding_dim, label_dim, num_layers,
            assign_hidden_dim, assign_ratio=assign_ratio, num_pooling=num_pooling, bn=bn, dropout=dropout,
            linkpred=linkpred, assign_input_dim=assign_input_dim, args=args)
        self.pred_input_dim_ast = self.pred_input_dim*(self.num_pooling+1)
        pre_input_dim = self.pred_input_dim_ast
        self.pred_model = self.build_pred_layers(pre_input_dim, pred_hidden_dims, label_dim, num_aggs=self.num_aggs)

        self.criterion = nn.BCEWithLogitsLoss(pos_weight=weight["siamese"])

        self.focal_loss = Focal_loss(weight["classifier"])

    def forward(self, f1, f2):
        fn_feat = []
        for fn in [f1, f2]:
            slice_feats = []
            for s in fn:
                x_ast, adj_ast, batch_num_nodes_ast, assign_ast = s
                ast_output = self.ast_module(x_ast, adj_ast, batch_num_nodes_ast, assign_x=assign_ast)
                output = ast_output
                                               
                # batch_num_nodes_ast :(batch-size, ); '0' means the slice is a padded slice.
                for i in range(len(batch_num_nodes_ast)):
                    if batch_num_nodes_ast[i] == 0:
                        output[i] = torch.zeros(self.pred_input_dim_ast)
                                                   
                slice_feats.append(output)

            fn_feat.append( torch.stack(slice_feats).sum(dim=0) )

        similarity_pre = F.cosine_similarity(fn_feat[0], fn_feat[1])

        author_pre = self.pred_model(fn_feat[0])

        return similarity_pre, author_pre

    def loss1(self, preds, labels):
        # for similarity prediction loss value
        loss = self.criterion(preds, labels)
        return loss

    def loss2(self, preds, labels):
        # for author prediction loss value
        # loss = super(SiameseArch_fn, self).loss(preds, labels)
        # loss = F.cross_entropy(preds, labels) 
        loss = self.focal_loss(preds, labels)
        return loss


class SiameseArch_RF_fn(SiameseArch_fn):
    """docstring for SiameseArch_RF_fn"""
    def __init__(self, max_num_nodes, input_dim_dict, hidden_dim, embedding_dim, label_dim, num_layers,
            assign_hidden_dim, assign_input_dim_dict, assign_ratio=0.25, assign_num_layers=-1, num_pooling=1,
            pred_hidden_dims=[50], concat=True, bn=True, dropout=0.0, linkpred=True, args=None):
        super(SiameseArch_RF_fn, self).__init__(max_num_nodes, input_dim_dict, hidden_dim, embedding_dim, label_dim, num_layers,
            assign_hidden_dim, assign_input_dim_dict, assign_ratio=assign_ratio, assign_num_layers=assign_num_layers, num_pooling=num_pooling,
            pred_hidden_dims=pred_hidden_dims, concat=concat, bn=bn, dropout=dropout, linkpred=linkpred, args=args)
        self.pred_model = None
    
    def get_fn_feature_vector(self, fn):
        slice_feats = []
        for s in fn:
            x_ast, adj_ast, batch_num_nodes_ast, assign_ast, x_pdg, adj_pdg, batch_num_nodes_pdg, assign_pdg = s

            ast_output = self.ast_module(x_ast, adj_ast, batch_num_nodes_ast, assign_x=assign_ast)
            pdg_output = self.pdg_module(x_pdg, adj_pdg, batch_num_nodes_pdg, assign_x=assign_pdg)
            output = torch.add(ast_output, pdg_output)

            # batch_num_nodes_ast :(batch-size, ); '0' means the slice is a padded slice.
            for i in range(len(batch_num_nodes_ast)):
                if batch_num_nodes_ast[i] == 0:
                    # output_list[i] = torch.zeros(self.pred_input_dim_ast)
                    output[i] = torch.zeros(self.pred_input_dim_ast)
                # print("type of output[{}]: {}\tvalue:{}".format(i,type(output_list[i]), output[i]))
            slice_feats.append(output)

        fn_feat = ( torch.stack(slice_feats).sum(dim=0) )
        return fn_feat

    def forward(self, f1, f2, phase="training"):
        if phase == "training":
            fn_feat = []
            for fn in [f1, f2]:
                fn_feat_vector = self.get_fn_feature_vector(fn)

                fn_feat.append( fn_feat_vector )

            similarity_pre = F.cosine_similarity(fn_feat[0], fn_feat[1])
            ret = similarity_pre
        elif phase == "eval":
            ret = self.get_fn_feature_vector(f1)
        else:
            ret = None
            print("wrong phase argument")

        return ret

    def loss(self, preds, labels):
        # for similarity prediction loss value
        loss = self.criterion(preds, labels)
        return loss

class SiameseDataset(Dataset):
    """docstring for SiameseDataset
    Sample graphs and nodes in graph
    """
    def __init__(self, slices_pairs_index_list, slices_graphs, features='default', normalize=True, assign_feat='default', max_num_nodes=0):
        # self.dataset = slices_pairs_index_list
        # self.dataset_len = len(slices_pairs_index_list)
        # self.slices_graphs = slices_graphs
        # self.slices_graphs_np = []
        # feat_dim_dict = dict()
        # astPath, pdgPath = slices_graphs[0]
        # feat_dim_dict['ast'] = int(splitext(basename(astPath))[0].split('_')[-3])
        # feat_dim_dict['pdg'] = int(splitext(basename(pdgPath))[0].split('_')[-3])
        # self.feat_dim = feat_dim_dict
        # self.assign_feat_dim = self.feat_dim

        # if max_num_nodes != 0:
        #     # ast and pdg use the same max_num_nodes from args
        #     self.max_num_nodes = max_num_nodes
        # else:
        #     self.max_num_nodes = args.max_nodes
        # self.pdg_adj_padded = np.zeros((self.max_num_nodes, self.max_num_nodes))
        # self.ast_adj_padded = np.zeros((self.max_num_nodes, self.max_num_nodes))

        # self.loadedGraphDict = self.loadGraphs()

        # self.hit = [0, 0, 0, 0]
        # self.takeNum = 0
        self.dataset = slices_pairs_index_list
        self.dataset_len = len(slices_pairs_index_list)
        self.slices_graphs_np = []
        self.max_num_nodes = max_num_nodes
        G_ast, G_pdg = slices_graphs[0]
        self.feat_dim = {'ast':G_ast.graph['feat_dim'], 'pdg':G_pdg.graph['feat_dim']}
        self.assign_feat_dim = self.feat_dim

        for s in slices_graphs:
            G_ast, G_pdg = s
            graph_np_dict = dict()
            adj = np.array(nx.to_numpy_matrix(G_ast))
            num_nodes = adj.shape[0]
            adj_padded = np.zeros((self.max_num_nodes, self.max_num_nodes))
            adj_padded[:num_nodes, :num_nodes] = adj
            graph_np_dict['ast_adj'] = adj_padded
            graph_np_dict['ast_nodes'] = num_nodes
            ast_feat = np.zeros((self.max_num_nodes, self.feat_dim['ast']), dtype=float)
            for i, u in enumerate(G_ast.nodes()):
                ast_feat[i, :] = util.node_dict(G_ast)[u]['feat']
            graph_np_dict['ast_feat'] = ast_feat
            graph_np_dict['ast_assign_feat'] = ast_feat

            adj = np.array(nx.to_numpy_matrix(G_pdg))
            num_nodes = adj.shape[0]
            adj_padded = np.zeros((self.max_num_nodes, self.max_num_nodes))
            adj_padded[:num_nodes, :num_nodes] = adj
            graph_np_dict['pdg_adj'] = adj_padded
            graph_np_dict['pdg_nodes'] = num_nodes
            pdg_feat = np.zeros((self.max_num_nodes, self.feat_dim['pdg']), dtype=float)
            for i, u in enumerate(G_pdg.nodes()):
                pdg_feat[i, :] = util.node_dict(G_pdg)[u]['feat']
            graph_np_dict['pdg_feat'] = pdg_feat
            graph_np_dict['pdg_assign_feat'] = pdg_feat
            graph_np_dict['label'] = G_ast.graph['label']
            self.slices_graphs_np.append(graph_np_dict)

    def __len__(self):
        return self.dataset_len

    def loadGraphs(self):
        # print(self.slices_graphs[indx])
        allSlicesIndx = []
        # for e in self.slices_graphs[:len(self.slices_graphs)//2]:
        for sp in self.dataset:
            s1_indx, s2_indx = sp
            allSlicesIndx.append(s1_indx)
            allSlicesIndx.append(s2_indx)

        freqDict = dict()
        for e in set(allSlicesIndx):
            count = allSlicesIndx.count(e)
            freqDict[e]=count

        l = sorted(list(freqDict.values()), reverse=True)
        mid = l[len(l)//2]
        top = [e for e in freqDict.keys() if freqDict[e] >= mid]
        print(mid, len(l), len(top))

        # load top x% frequence graphs
        p2GraphDict = dict()
        for s_indx in top:
            ast_p, pdg_p = self.slices_graphs[s_indx]
            p2GraphDict[ast_p] = nx.read_gpickle(ast_p)
            p2GraphDict[pdg_p] = nx.read_gpickle(pdg_p)

        for e in set(allSlicesIndx):
            ast_p, pdg_p = self.slices_graphs[e]
            if ast_p not in p2GraphDict.keys():
                p2GraphDict[ast_p] = None
            if pdg_p not in p2GraphDict.keys():
                p2GraphDict[pdg_p] = None

        return p2GraphDict

        # ast = nx.read_gpickle(self.slices_graphs[indx][0])
        # pdg = nx.read_gpickle(self.slices_graphs[indx][1])
        # return (ast, pdg)

    def __getitem__(self, idx):
        # # print("get element in dataset")
        # output = []
        # s1_index, s2_index = self.dataset[idx]
        # # s1_ast_pdg = self.loadGraphs(s1_index)
        # # s2_ast_pdg = self.loadGraphs(s2_index)
        # s1_ast_p, s1_pdg_p = self.slices_graphs[s1_index]
        # s1_ast, s1_pdg = (self.loadedGraphDict[s1_ast_p], self.loadedGraphDict[s1_pdg_p])
        # if s1_ast == None:
        #     s1_ast = nx.read_gpickle(s1_ast_p)
        #     self.hit[0] += 1
        # if s1_pdg == None:
        #     s1_pdg = nx.read_gpickle(s1_pdg_p)
        #     self.hit[1] += 1

        # s2_ast_p, s2_pdg_p = self.slices_graphs[s2_index]        
        # s2_ast, s2_pdg = (self.loadedGraphDict[s2_ast_p], self.loadedGraphDict[s2_pdg_p])
        # if s2_ast == None:
        #     s2_ast = nx.read_gpickle(s2_ast_p)
        #     self.hit[2] += 1
        # if s2_pdg == None:
        #     s2_pdg = nx.read_gpickle(s2_pdg_p)
        #     self.hit[3] += 1
        # # s1_ast_pdg = (self.loadedGraphDict[self.slices_graphs[s1_index][0]], self.loadedGraphDict[self.slices_graphs[s1_index][1]])
        # # s2_ast_pdg = (self.loadedGraphDict[self.slices_graphs[s2_index][0]], self.loadedGraphDict[self.slices_graphs[s2_index][1]])
        # s1_ast_pdg = (s1_ast, s1_pdg)
        # s2_ast_pdg = (s2_ast, s2_pdg)
        # for s in [s1_ast_pdg, s2_ast_pdg]:
        #     G_ast, G_pdg = s
        #     for i, g in enumerate(list(s)):
        #         max_rows = self.max_num_nodes
        #         if i == 0:
        #             adj_padded = self.ast_adj_padded.copy()
        #         else:
        #             adj_padded = self.pdg_adj_padded.copy()
        #         graph_np_dict = dict()
        #         adj = np.array(nx.to_numpy_matrix(g))
        #         num_nodes = adj.shape[0]
        #         adj_padded[:num_nodes, :num_nodes] = adj
        #         graph_np_dict['adj'] = adj_padded
        #         graph_np_dict['num_nodes'] = num_nodes
        #         feat = np.zeros((max_rows, g.graph['feat_dim']), dtype=float)
        #         for j, u in enumerate(g.nodes()):
        #             feat[j, :] = util.node_dict(g)[u]['feat']
        #         graph_np_dict['feats'] = feat.copy()
        #         # print(graph_np_dict['feats'].shape)
        #         graph_np_dict['assign_feats'] = graph_np_dict['feats']
        #         graph_np_dict['label'] = g.graph['label']
        #         output.append(graph_np_dict)

        # self.takeNum += 1
        # if self.takeNum % 10 == 0:
        #     print(self.hit)
        #     self.hit = [0, 0, 0, 0]
        # if output[0]['label'] == output[2]['label']:
        #     similarity_label = 1
        # else:
        #     similarity_label = 0
        # author_label = output[0]['label']
        # # print(author_label)
        # s1_ast, s1_pdg, s2_ast, s2_pdg = tuple(output)
        # return s1_ast, s1_pdg, s2_ast, s2_pdg, similarity_label, author_label
        s1_index, s2_index = self.dataset[idx]
        s1_ast = {'adj':self.slices_graphs_np[s1_index]['ast_adj'],
                'feats': self.slices_graphs_np[s1_index]['ast_feat'].copy(),
                'num_nodes':self.slices_graphs_np[s1_index]['ast_nodes'],
                'assign_feats':self.slices_graphs_np[s1_index]['ast_assign_feat'].copy()}
        s1_pdg = {'adj':self.slices_graphs_np[s1_index]['pdg_adj'],
                'feats': self.slices_graphs_np[s1_index]['pdg_feat'].copy(),
                'num_nodes':self.slices_graphs_np[s1_index]['pdg_nodes'],
                'assign_feats':self.slices_graphs_np[s1_index]['pdg_assign_feat'].copy()}
        s2_ast = {'adj':self.slices_graphs_np[s2_index]['ast_adj'],
                'feats': self.slices_graphs_np[s2_index]['ast_feat'].copy(),
                'num_nodes':self.slices_graphs_np[s2_index]['ast_nodes'],
                'assign_feats':self.slices_graphs_np[s2_index]['ast_assign_feat'].copy()}
        s2_pdg = {'adj':self.slices_graphs_np[s2_index]['pdg_adj'],
                'feats': self.slices_graphs_np[s2_index]['pdg_feat'].copy(),
                'num_nodes':self.slices_graphs_np[s2_index]['pdg_nodes'],
                'assign_feats':self.slices_graphs_np[s2_index]['pdg_assign_feat'].copy()}
        if self.slices_graphs_np[s1_index]['label'] == self.slices_graphs_np[s2_index]['label']:
            similarity_label = 1
        else:
            similarity_label = 0
        author_label = self.slices_graphs_np[s1_index]['label']

        return s1_ast, s1_pdg, s2_ast, s2_pdg, similarity_label, author_label

class SiameseDataset_fn(Dataset):
    """docstring for SiameseDataset_fn"""
    # def __init__(self, fn_pairs_index_list, slices_graphs, features='default', normalize=True, assign_feat='default', max_num_nodes=0):
    def __init__(self, fn_pairs_index_list, fn_list, slices_graphs, max_num_nodes=0):
        super(SiameseDataset_fn, self).__init__()
        self.dataset = fn_pairs_index_list
        self.fns_list = fn_list
        self.dataset_len = len(fn_pairs_index_list)
        self.slices = slices_graphs
        if max_num_nodes != 0:
            # ast and pdg use the same max_num_nodes from args
            self.max_num_nodes = max_num_nodes
        else:
            self.max_num_nodes = args.max_nodes
        self.pdg_adj_padded = np.zeros((self.max_num_nodes, self.max_num_nodes))
        self.ast_adj_padded = np.zeros((self.max_num_nodes, self.max_num_nodes))

        feat_dim_dict = dict()
        g_ast, g_pdg = slices_graphs[0]
        feat_dim_dict['ast'] = g_ast.graph['feat_dim']
        feat_dim_dict['pdg'] = g_pdg.graph['feat_dim']
        self.feat_dim = feat_dim_dict 
        self.assign_feat_dim = self.feat_dim

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        fn1_idx, fn2_idx = self.dataset[idx]
        fn1_slices = self.fns_list[fn1_idx]
        fn2_slices = self.fns_list[fn2_idx]
        # print(fn1_slices, fn2_slices)

        fn1 = dict()
        # print(fn1_slices)
        fn1['slices'] = []
        for index in fn1_slices:
            # fn1["slice"+str(index)] = dict()
            if index == -1:
                s_dict['ast']['num_nodes'] = 0
                s_dict['pdg']['num_nodes'] = 0
                fn1['slices'].append(s_dict)
                # print("\tmet '-1'")
                continue
            s_dict = dict()
            for i, g in enumerate(list(self.slices[index])):
                # each slice contains a 'ast' and a 'pdg'
                max_rows = self.max_num_nodes
                if i == 0:
                    key = 'ast'
                    adj_padded = self.ast_adj_padded.copy()
                else:
                    key = 'pdg'
                    adj_padded = self.pdg_adj_padded.copy()
                graph_np_dict = dict()
                adj = np.array(nx.to_numpy_matrix(g))
                num_nodes = adj.shape[0]
                adj_padded[:num_nodes, :num_nodes] = adj
                graph_np_dict['adj'] = adj_padded
                graph_np_dict['num_nodes'] = num_nodes
                feat = np.zeros((max_rows, g.graph['feat_dim']), dtype=float)
                for j, u in enumerate(g.nodes()):
                    feat[j, :] = util.node_dict(g)[u]['feat']
                graph_np_dict['feats'] = feat.copy()
                # print(graph_np_dict['feats'].shape)
                graph_np_dict['assign_feats'] = graph_np_dict['feats']
                # fn1["slice"+str(index)][key] = graph_np_dict
                s_dict[key] = graph_np_dict
                fn1_label = g.graph['label']
            # print("\tconstruct dict")
            fn1['slices'].append(s_dict)
        # print("*"*5+"fn1[slices] len = {}".format(len(fn1['slices']))+"\t"+"*"*5)

        fn2 = dict()
        fn2['slices'] = []
        for index in fn2_slices:
            if index == -1:
                s_dict['ast']['num_nodes'] = 0
                s_dict['pdg']['num_nodes'] = 0
                fn2['slices'].append(s_dict)
                continue
            # fn2["slice"+str(index)] = dict()
            s_dict = dict()
            for i, g in enumerate(list(self.slices[index])):
                # each slice contains a 'ast' and a 'pdg'
                max_rows = self.max_num_nodes
                if i == 0:
                    key = 'ast'
                    adj_padded = self.ast_adj_padded.copy()
                else:
                    key = 'pdg'
                    adj_padded = self.pdg_adj_padded.copy()
                graph_np_dict = dict()
                adj = np.array(nx.to_numpy_matrix(g))
                num_nodes = adj.shape[0]
                adj_padded[:num_nodes, :num_nodes] = adj
                graph_np_dict['adj'] = adj_padded
                graph_np_dict['num_nodes'] = num_nodes
                feat = np.zeros((max_rows, g.graph['feat_dim']), dtype=float)
                for j, u in enumerate(g.nodes()):
                    feat[j, :] = util.node_dict(g)[u]['feat']
                graph_np_dict['feats'] = feat.copy()
                # print(graph_np_dict['feats'].shape)
                graph_np_dict['assign_feats'] = graph_np_dict['feats']
                # fn2["slice"+str(index)][key] = graph_np_dict
                s_dict[key] = graph_np_dict
                fn2_label = g.graph['label']
            fn2['slices'].append(s_dict)
        # print("*"*5+"fn2[slices] len = {}".format(len(fn2['slices']))+"\t"+"*"*5)

        if fn1_label == fn2_label:
            similarity_label = 1
        else:
            similarity_label = 0
        author_label = fn1_label
        # print(author_label)
        # print(len(fn1['slices']), len(fn2['slices']))
        return fn1, fn2, similarity_label, author_label
        
class SiameseDataset_RF_fn(SiameseDataset_fn):
    """docstring for SiameseDataset_RF_fn"""
    def __init__(self, fn_pairs_index_list, fn_list, slices_graphs, max_num_nodes):
        super(SiameseDataset_RF_fn, self).__init__(fn_pairs_index_list, fn_list, slices_graphs, max_num_nodes)
        
class SiameseDataset_ast_pdg_fn(Dataset):
    """docstring for SiameseDataset_ast_pdg_fn
    only use AST or PDG to represent a slice but keep the form of a slice
    (ast_graph, 0) or (pdg_graph, 0)
    """
    # def __init__(self, fn_pairs_index_list, slices_graphs, features='default', normalize=True, assign_feat='default', max_num_nodes=0):
    def __init__(self, fn_pairs_index_list, fn_list, slices_graphs, max_num_nodes=0):
        super(SiameseDataset_ast_pdg_fn, self).__init__()
        self.dataset = fn_pairs_index_list
        self.fns_list = fn_list
        self.dataset_len = len(fn_pairs_index_list)
        self.slices = slices_graphs
        if max_num_nodes != 0:
            # ast and pdg use the same max_num_nodes from args
            self.max_num_nodes = max_num_nodes
        else:
            self.max_num_nodes = args.max_nodes
        # self.pdg_adj_padded = np.zeros((self.max_num_nodes, self.max_num_nodes))
        # self.ast_adj_padded = np.zeros((self.max_num_nodes, self.max_num_nodes))
        self.ast_adj_padded = np.zeros((self.max_num_nodes, self.max_num_nodes))

        feat_dim_dict = dict()
        g_ast, _ = slices_graphs[0]
        feat_dim_dict['ast'] = g_ast.graph['feat_dim']
        # feat_dim_dict['pdg'] = g_pdg.graph['feat_dim']
        self.feat_dim = feat_dim_dict 
        self.assign_feat_dim = self.feat_dim

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        fn1_idx, fn2_idx = self.dataset[idx]
        fn1_slices = self.fns_list[fn1_idx]
        fn2_slices = self.fns_list[fn2_idx]
        # print(fn1_slices, fn2_slices)

        fn1 = dict()
        # print(fn1_slices)
        fn1['slices'] = []
        for index in fn1_slices:
            # fn1["slice"+str(index)] = dict()
            if index == -1:
                s_dict['ast']['num_nodes'] = 0
                # s_dict['pdg']['num_nodes'] = 0
                fn1['slices'].append(s_dict)
                # print("\tmet '-1'")
                continue
            s_dict = dict()
            for i, g in enumerate(list(self.slices[index])):
                # each slice contains a 'ast' and a 'pdg'
                max_rows = self.max_num_nodes
                if i == 0:
                    key = 'ast'
                    adj_padded = self.ast_adj_padded.copy()
                else:
                    # use 'ast' to contain AST or PDG
                    continue
                    key = 'pdg'
                    adj_padded = self.pdg_adj_padded.copy()
                graph_np_dict = dict()
                adj = np.array(nx.to_numpy_matrix(g))
                num_nodes = adj.shape[0]
                adj_padded[:num_nodes, :num_nodes] = adj
                graph_np_dict['adj'] = adj_padded
                graph_np_dict['num_nodes'] = num_nodes
                feat = np.zeros((max_rows, g.graph['feat_dim']), dtype=float)
                for j, u in enumerate(g.nodes()):
                    feat[j, :] = util.node_dict(g)[u]['feat']
                graph_np_dict['feats'] = feat.copy()
                # print(graph_np_dict['feats'].shape)
                graph_np_dict['assign_feats'] = graph_np_dict['feats']
                # fn1["slice"+str(index)][key] = graph_np_dict
                s_dict[key] = graph_np_dict
                fn1_label = g.graph['label']
            # print("\tconstruct dict")
            fn1['slices'].append(s_dict)
        # print("*"*5+"fn1[slices] len = {}".format(len(fn1['slices']))+"\t"+"*"*5)

        fn2 = dict()
        fn2['slices'] = []
        for index in fn2_slices:
            if index == -1:
                s_dict['ast']['num_nodes'] = 0
                # s_dict['pdg']['num_nodes'] = 0
                fn2['slices'].append(s_dict)
                continue
            # fn2["slice"+str(index)] = dict()
            s_dict = dict()
            for i, g in enumerate(list(self.slices[index])):
                # each slice contains a 'ast' and a 'pdg'
                max_rows = self.max_num_nodes
                if i == 0:
                    key = 'ast'
                    adj_padded = self.ast_adj_padded.copy()
                else:
                    continue
                    key = 'pdg'
                    adj_padded = self.pdg_adj_padded.copy()
                graph_np_dict = dict()
                adj = np.array(nx.to_numpy_matrix(g))
                num_nodes = adj.shape[0]
                adj_padded[:num_nodes, :num_nodes] = adj
                graph_np_dict['adj'] = adj_padded
                graph_np_dict['num_nodes'] = num_nodes
                feat = np.zeros((max_rows, g.graph['feat_dim']), dtype=float)
                for j, u in enumerate(g.nodes()):
                    feat[j, :] = util.node_dict(g)[u]['feat']
                graph_np_dict['feats'] = feat.copy()
                # print(graph_np_dict['feats'].shape)
                graph_np_dict['assign_feats'] = graph_np_dict['feats']
                # fn2["slice"+str(index)][key] = graph_np_dict
                s_dict[key] = graph_np_dict
                fn2_label = g.graph['label']
            fn2['slices'].append(s_dict)
        # print("*"*5+"fn2[slices] len = {}".format(len(fn2['slices']))+"\t"+"*"*5)

        if fn1_label == fn2_label:
            similarity_label = 1
        else:
            similarity_label = 0
        author_label = fn1_label
        # print(author_label)
        # print(len(fn1['slices']), len(fn2['slices']))
        return fn1, fn2, similarity_label, author_label
