#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from collections import defaultdict
import pandas as pd
import numpy as np


# def dfs(graph, start, visited=None):
#    if visited is None:
#        visited = set()
#    visited.add(start)
#    for next in graph[start] - visited:
#        dfs(graph, next, visited)
#    return visited

def dfs(graph):
    visited = set()
    visited_list = list()
    for node in list(graph):
        if node in visited:
            continue
        stack = [node]

        while stack:
            vertex = stack.pop()
            if vertex not in visited:
                visited.add(vertex)
                visited_list.append(vertex)
                stack.extend(graph[vertex] - visited)
    return visited_list


max_ = -float('inf')
operationDic = defaultdict(int)
# np.save('operationDic.npy', operationDic)
counter = 1
name, corpus = [], []
fea_num = 10000

for i in range(0, 1000):

    path = 'path' + str(i) + '/'

    c = 0
    folderNames = os.listdir(path)
    operationDic = np.load('operationDic.npy', allow_pickle='TRUE').item()

    for folder in folderNames:
        c += 1

        if c % 1000 == 0:
            print(str(i) + ': ' + str(c))

        folderName = path + folder
        if ".hex" not in folderName: continue  # check can remove
        operation_file = folderName + '/op.facts'
        print(operation_file)

        opDic = defaultdict()
        codeDic = defaultdict(int)

        with open(operation_file, 'r') as op:
            for line in op.readlines():
                code = line.split('\t')[0]
                operation = line.split('\t')[1].strip()
                opDic[code] = operation
                if operation not in operationDic:
                    operationDic[operation] = counter
                    counter += 1

        edge_file = folderName + '/edge.facts'

        adjacency_matrix = defaultdict(set)
        adjacency_matrix2 = defaultdict(set)
        with open(edge_file, 'r') as edg:
            for line in edg.readlines():
                try:
                    code1 = line.split('\t')[0].strip()
                    code2 = line.split('\t')[1].strip()
                    adjacency_matrix[codeDic[code1]].add(codeDic[code2])
                    adjacency_matrix2[code1].add(code2)
                except:
                    pass

        try:
            l = dfs(adjacency_matrix2)
            #             dfs_l = [opDic[code] for code in l[:fea_num]]
            dfs_l = [opDic[code] for code in l]
            #             dfs_o = [operationDic[operation] for operation in dfs_l]
            dfs_o = dfs_l
            name.append(folder)
            corpus.append(dfs_o)
        except:
            pass

    dfName = pd.DataFrame(name, columns=['Name'])

    new_corpus = corpus
    corpdf = pd.DataFrame(new_corpus)
    data1 = pd.concat([dfName, corpdf], axis=1)
    data1.to_csv('path' + str(i) + '.csv', index=False,
                 header=False)

