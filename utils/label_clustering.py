import pandas as pd
from matplotlib import pyplot as plt
import os
from sklearn.cluster import AgglomerativeClustering
import numpy as np
from treelib import Tree
import pickle

def build_tree(tree, linkage_dict, node_id, labels, offset=0, parent=None):
    if node_id < len(labels):  # It's a leaf node (original data point)]
        if not isinstance(parent, int):
            tree.create_node(f"{labels[node_id]}", f"{labels[node_id]}", parent=parent)
        else:
            tree.create_node(f"{labels[node_id]}", f"{labels[node_id]}", parent=parent+offset)
    else:  # It's a merged cluster
        left, right = int(linkage_dict[node_id][0]), int(linkage_dict[node_id][1])
        if not isinstance(parent, int):
            tree.create_node(f"o", node_id+offset, parent=parent)
        else:
            tree.create_node(f"o", node_id+offset, parent=parent+offset)
        build_tree(tree, linkage_dict, left, labels, offset=offset, parent=node_id)
        build_tree(tree, linkage_dict, right, labels, offset=offset, parent=node_id)

def build_entity_tree_from_human_perception(folder_path, output_path):
    
    sim_mats = []
    polarity_arrays = []
    prev_columns = []
    prev_rows = []
    for f in os.listdir(folder_path):

        polarity_df = pd.read_excel(os.path.join(folder_path, f),sheet_name='Sheet2',index_col=0)
        polarity_arrays.append(polarity_df.iloc[:,0].values)
        
        df = pd.read_excel(os.path.join(folder_path, f),sheet_name='Sheet1',index_col=0)
        df = df.fillna(0)
        sim_values = df.values

        columns = df.columns.to_list()
        rows = df.index.to_list()
        if sim_values.shape[0] !=sim_values.shape[1]:
            sim_values = sim_values[:,1:]
            columns = columns[1:]
        
        if len(prev_columns) == 0:
            pass
        else:
            assert prev_columns == columns
            assert prev_rows == rows
        prev_columns = columns
        prev_rows = rows
        sim_values = sim_values/10
        sim_values = sim_values + sim_values.T
        np.fill_diagonal(sim_values,1)
        sim_mats.append(sim_values)

    polarity_mat = np.array(polarity_arrays)
    polarity_list = np.array([np.bincount(polarity_mat[:,i]+1).argmax()-1 for i in range(polarity_mat.shape[-1])])

    ids = [np.where(polarity_list %2 == 0)[0], np.where(polarity_list == 1)[0], np.where(polarity_list == -1)[0]]
    sim_mat_average =np.array(sim_mats).mean(axis=0)

    tree = Tree()
    tree.create_node('root', 'root')
    cumulative_offset = 0
    for _index, _id in enumerate(ids):
        tree.create_node(f"Main Class {_index}", f"Main Class {_index}",parent='root')
        polarity_submat = sim_mat_average[_id,:][:,_id]
        clustering = AgglomerativeClustering(metric='precomputed',linkage='complete').fit(1-polarity_submat)
        linkage = dict(enumerate(clustering.children_, clustering.n_leaves_))
        sub_columns = df.columns[_id]
        build_tree(tree, linkage, len(linkage)+len(sub_columns)-1, sub_columns,offset=cumulative_offset, parent=f"Main Class {_index}")
        cumulative_offset += len(linkage)+len(sub_columns)

    pickle.dump(tree, open(output_path, 'wb'))
    # tree.save2file(output_path)
    # return tree



def get_distance(leaf_label_1, leaf_label_2, tree, max_depth=5):

    node_1 = tree.get_node(leaf_label_1)
    node_2 = tree.get_node(leaf_label_2)

    if (node_1 is None) or (node_2 is None):
        return 1

    if leaf_label_1 == leaf_label_2:
        return 0

    if tree.depth(node_1) > max_depth:
        for _ in range(tree.depth(node_1) - max_depth):
            node_1 = tree.parent(node_1.identifier)

    if tree.depth(node_2) > max_depth:
        for _ in range(tree.depth(node_2) - max_depth):
            node_2 = tree.parent(node_2.identifier)
    
    if tree.depth(node_1) > tree.depth(node_2):
        distance = tree.depth(node_1) - tree.depth(node_2)
        for _ in range(distance):
            node_1 = tree.parent(node_1.identifier)
    else:
        distance = tree.depth(node_2) - tree.depth(node_1)
        for _ in range(distance):
            node_2 = tree.parent(node_2.identifier)
    distance = distance + 1
    while(node_1 != node_2):
        distance = distance + 1
        node_1 = tree.parent(node_1.identifier)
        node_2 = tree.parent(node_2.identifier)
    return distance/(max_depth + 1)