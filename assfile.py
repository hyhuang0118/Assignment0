import numpy as np
import pandas as pd
import copy
import itertools
import os
import random


def importCSV(path_to_data):
    df = pd.read_csv(path_to_data, dtype='float')
    df = df.drop(['veil-type_p'], axis=1)
    if 'class' in list(df.columns.values):
        df['class'].replace(0, -1, inplace=True)
    return df


def cut_tree(tree, node, features, y, depth, m, mode='DT'):
    continue_tf_1 = True
    continue_tf_0 = True
    if tree[node]['continue'] == True:
        if node == 'root':
            children = ['1', '0']
        else:
            children = [node + '-1', node + '-0']
        m_Count=0
        sel_B = 0
        sel_feature = features[0]
        for feature in features:
            if feature not in tree[node]['feature_path']:  # check feature used or not.
                temp_data = tree[node]['data']
                D_1 = temp_data.loc[temp_data[feature] == 1]
                D_0 = temp_data.loc[temp_data[feature] == 0]
                # counting number of features with y=1, -1
                c1_1 = len(D_1.loc[D_1[y] == 1])  
                c1_0 = len(D_1.loc[D_1[y] == -1])
                # counting number of features with y=1, -1
                c0_1 = len(D_0.loc[D_0[y] == 1])  
                c0_0 = len(D_0.loc[D_0[y] == -1])
                #p is the probabilty that y = 1 or 0
                p1 = len(D_1) / len(temp_data)
                p0 = len(D_0) / len(temp_data)
                if mode == 'adaboost':
                    sum1 = (D_1.sum()['D'])
                    sum0 = (D_0.sum()['D'])
                    sum_temp = (temp_data.sum()['D'])
                    p1 = (sum1 / (sum_temp))
                    p0 = (sum0 / (sum_temp))

                if (c1_1 == 0) and (c1_0 == 0):
                    p1_1 = 0
                    p1_0 = 0
                elif mode == 'adaboost':
                    sumD1 = (D_1.loc[D_1[y] == 1].sum()['D'])
                    sumD0 = (D_1.loc[D_1[y] == -1].sum()['D'])
                    p1_1 = (sumD1 / (sumD1 + sumD0))
                    p1_0 = (sumD0 / (sumD1 + sumD0))
                else:
                    p1_1 = (c1_1 / (c1_1 + c1_0))
                    p1_0 = (c1_0 / (c1_1 + c1_0))

                if (c0_1 == 0) and (c0_0 == 0):
                    p0_1 = 0
                    p0_0 = 0
                elif mode == 'adaboost':
                    sumD1 = (D_0.loc[D_0[y] == 1].sum()['D'])
                    sumD0 = (D_0.loc[D_0[y] == -1].sum()['D'])
                    p0_1 = (sumD1 / (sumD1 + sumD0))
                    p0_0 = (sumD0 / (sumD1 + sumD0))
                else:
                    p0_1 = (c0_1 / (c0_1 + c0_0))
                    p0_0 = (c0_0 / (c0_1 + c0_0))

                U_1 = 1 - (p1_1 ** 2) - (p1_0 ** 2)
                U_0 = 1 - (p0_1 ** 2) - (p0_0 ** 2)
                B = tree[node]['U'] - p1 * U_1 - p0 * U_0
                if B >= sel_B:
                    continue_tf_1 = True
                    continue_tf_0 = True

                    sel_B = B
                    sel_feature = feature
                    #count of 0s and 1s
                    tree[children[0]]['f=1'] = c1_1
                    tree[children[0]]['f=0'] = c1_0
                    tree[children[0]]['data'] = D_1
                    tree[children[0]]['prob'] = p1
                    tree[children[0]]['p1'] = p1_1
                    tree[children[0]]['p0'] = p1_0
                    tree[children[0]]['U'] = U_1         

                    tree[children[1]]['f=1'] = c0_1
                    tree[children[1]]['f=0'] = c0_0
                    tree[children[1]]['data'] = D_0
                    tree[children[1]]['prob'] = p0
                    tree[children[1]]['p1'] = p0_1
                    tree[children[1]]['p0'] = p0_0
                    tree[children[1]]['U'] = U_0
                    
                    if (c1_1 == 0) or (c1_0 == 0):
                        continue_tf_1 = False

                    if (c0_1 == 0) or (c0_0 == 0):
                        continue_tf_0 = False
                m_Count=m_Count+1
                if m_Count == m:
                    break
        if node == 'root':
            tree[node]['feature_path'].append(sel_feature)
            tree[node]['split_on'] = sel_feature
        else:
            if len(node.split('-')) == 1:
                Parent_Node = 'root'
            else:
                Parent_Node = node.split('-')[:-1]
                Parent_Node = '-'.join(Parent_Node)

            tree[node]['feature_path'] = tree[Parent_Node]['feature_path'][:]
            tree[node]['feature_path'].append(sel_feature)
            tree[node]['split_on'] = sel_feature

    return tree, continue_tf_1, continue_tf_0


def b_nu_tree(data, features, y, depth, mode):
    c0 = len(data.loc[data[y] == -1])
    c1 = len(data.loc[data[y] == 1])

    if mode == 'adaboost':
        sumD1 = sum(data.loc[data[y] == 1]['D'])
        sumD0 = sum(data.loc[data[y] == -1]['D'])
        p1 = (sumD1 / (sumD1 + sumD0))
        p0 = (sumD0 / (sumD1 + sumD0))
    else:
        p1 = (c1 / (c1 + c0))
        p0 = (c0 / (c1 + c0))

    tree = {}
    tree['root'] = {}
    tree['root']['data'] = data
    tree['root']['f=0'] = c0
    tree['root']['f=1'] = c1
    tree['root']['prob'] = 1
    tree['root']['U'] = 1 - p1 ** 2 - p0 ** 2
    tree['root']['continue'] = True
    tree['root']['split_on'] = None
    tree['root']['p1'] = p1
    tree['root']['p0'] = p0
    tree['root']['feature_path'] = []

    for i in range(depth):
        temp = list(itertools.product([1, 0], repeat=i + 1))

        if len(temp[0]) == 1:
            temp = [str(i[0]) for i in temp]
        else:
            temp = ['-'.join(map(str, i)) for i in temp]
        for ii in temp:
            tree[ii] = {}
            tree[ii]['data'] = None
            tree[ii]['f=0'] = None
            tree[ii]['f=1'] = None
            tree[ii]['prob'] = None
            tree[ii]['U'] = None
            tree[ii]['continue'] = True
            tree[ii]['split_on'] = None
            tree[ii]['p1'] = None
            tree[ii]['p0'] = None
            tree[ii]['feature_path'] = []

    return tree


def pt_t(tree):
    for key in tree.keys():
        print('key: {}'.format(key))
        print('f=0: {}'.format(tree[key]['f=0']))
        print('f=1: {}'.format(tree[key]['f=1']))
        print('prob: {}'.format(tree[key]['prob']))
        print('U: {}'.format(tree[key]['U']))
        print('continue: {}'.format(tree[key]['continue']))
        print('split_on: {}'.format(tree[key]['split_on']))
        print('p1: {}'.format(tree[key]['p1']))
        print('p0: {}'.format(tree[key]['p0']))
        print('feature_path: {}'.format(tree[key]['feature_path']))
        print('____________________________________________')


def study(data, features, y, depth, mode='DT', view_tree=False, m=10000000000000, bagging=0):
    tree = b_nu_tree(data, features, y, depth, mode)
    nodes = list(tree.keys())
    for node in nodes:
        if (len(node.split('-')) == depth) and (node != 'root'):
            continue
        if bagging==1:
            features = random.sample(features, len(features))
        tree, continue_tf_1, continue_tf_0 = cut_tree(tree, node, features, y, depth, m, mode)
        child = ''
        if continue_tf_0 == False:
            if node == 'root':
                child = '0'
            else:
                child = node + '-' + '0'
            children = []
            for node_i in tree.keys():
                split_node = node_i.split('-')
                len_node = len(child.split('-'))
                if len(split_node) > len_node:
                    if '-'.join(split_node[0:len_node]) == child:
                        children.append(node_i)
            all_children = children
            for child in all_children:
                tree[child]['continue'] = False
        if continue_tf_1 == False:
            if node == 'root':
                child = '1'
            else:
                child = node + '-' + '1'
            children = []
            for node_i in tree.keys():
                split_node = node_i.split('-')
                len_node = len(child.split('-'))
                if len(split_node) > len_node:
                    if '-'.join(split_node[0:len_node]) == child:
                        children.append(node_i)
            all_children = children
            for child in all_children:
                tree[child]['continue'] = False
    tree = {i: j for i, j in tree.items() if tree[i]['p1'] != None}

    if view_tree == True:
        pt_t(tree)

    return tree

def c_error(tree, data, y=None, mode='DT',y_pred=[]):
    error_count = 0
    errlog = []
    if mode == 'RF':
        for index, ex in data.iterrows():
            if y != None:
                if (y_pred[index] == ex[y]):
                    errlog.append(0)
                else:
                    errlog.append(1)
                    error_count += 1
        return error_count

    y_pred_tot = []
    for index, ex in data.iterrows():
        y_pred = rn_tree(tree, ex)
        y_pred_tot.append(y_pred)

        if y != None:
            if (y_pred == ex[y]):
                errlog.append(0)
            else:
                errlog.append(1)
                if mode == 'adaboost':
                    error_count += ex['D']
                else:
                    error_count += 1
        # i += 1
    return error_count, y_pred_tot

def rn_tree(tree, ex):
    node = 'root'
    feat = tree[node]['split_on']
    node = str(int(ex[feat]))
    if (tree[node]['split_on'] is None) or (tree[node]['split_on'] == "None"):
        if tree[node]['p1'] >= tree[node]['p0']:
            return 1
        else:
            return -1

    while (True):
        feat = tree[node]['split_on']

        if (tree[node]['split_on'] is None) or (tree[node]['split_on'] == "None"):
            if tree[node]['p1'] >= tree[node]['p0']:
                return 1
            else:
                return -1

        newNode = node + '-' + str(int(ex[feat]))
        if newNode not in tree.keys():
            if tree[node]['p1'] >= tree[node]['p0']:
                return 1
            else:
                return -1

        node = newNode
