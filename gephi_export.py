import sklearn.neighbors as NN
import networkx as nx
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd


def preprocess(df, n):
    lengths = np.array([])
    for c in df:
        if type(df[c][0]) is str:
            length = df[c].str.len().sum()
            lengths = np.append(lengths, length)

    largest = np.where(lengths == np.max(lengths))[0][0]
    data_name = df.columns[largest]
    l = list(df.columns)
    l.remove(data_name)
    l = np.append([data_name], l)
    df = df[l]
    rows = np.random.choice(df.index, size=n, replace=False)
    df = df.iloc[rows]
    df = df.reset_index()
    df = df.drop(columns="index")
    df[df.columns[0]] = df[df.columns[0]].str.strip()
    return(df)

def feature_extraction(file_name, model_name, n):
    print(file_name)
    df = pd.read_csv(file_name, encoding = "ISO-8859-1")
    df = preprocess(df, n)
    data, data_attributes = df[df.columns[0]], df[df.columns[1:]]

    feature_dict = {}
    mapping = {}
    attributes = {}
    for i, text in enumerate(data):
        feature_dict[i] = text
        mapping[i] = text
        attributes[mapping[i]] = dict(zip(list(data_attributes.columns), [attribute[i] for attribute in data_attributes.to_numpy().T]))
        print(attributes[mapping[i]])
    model = SentenceTransformer(model_name) # 'whaleloops/phrase-bert'
    feature_list = [text for features in feature_dict.values() for text in np.unique(features)]
    feature_extract = model.encode(feature_list, device="cuda")
    return(feature_extract, mapping, attributes)

def gephi(feature_extract, file_name, model_name, mapping, attributes):
    nn = NN.kneighbors_graph(feature_extract, n_neighbors=4, mode="distance", metric="l1")

    _ = np.exp(-nn.data**2/np.mean(nn.data)**2, out=nn.data)

    G = nx.DiGraph(nn)
    H = nx.relabel_nodes(G, mapping)
    nx.set_node_attributes(H, attributes)
    nx.write_gexf(H, file_name + "_" +  model_name + ".gexf")

def gephi_export(file_name, model_name, n):
    feature_extract, mapping, attributes = feature_extraction(file_name, model_name, n)
    file_name = file_name.replace(".csv", "")
    file_name = file_name.replace("data\\", "")
    model_name = model_name.replace("/", "_")
    gephi(feature_extract, file_name, model_name, mapping, attributes)

def run_all(datasets, models, n, graph=False):
    model_dict = {}
    for model in models:
        for i, dataset in enumerate(datasets):
            if i == 0:
                feature_extract = feature_extraction(dataset, model, n)[0]
                print(feature_extract.shape)
                if graph == True: gephi_export(dataset, model, n)
            else:
                feature_extract = np.dstack((feature_extract, feature_extraction(dataset, model, n)[0]))
                print(feature_extract.shape)
                if graph == True: gephi_export(dataset, model, n)
        model_dict[model] = feature_extract
    return(model_dict)
