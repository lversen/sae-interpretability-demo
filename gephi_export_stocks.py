import sklearn.neighbors as NN
import networkx as nx
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
import torch

def batch_encode(model, feature_list, batch_size=16):
    embeddings = []
    for i in range(0, len(feature_list), batch_size):
        batch = feature_list[i:i+batch_size]
        batch_embeddings = model.encode(batch, device='cuda')  # Specify device
        embeddings.extend(batch_embeddings)
        torch.cuda.empty_cache()  # Clear cache after each batch
        print(i)
    return np.array(embeddings)

def preprocess(df, n):
    lengths = np.array([])
    for c in df:
        if type(df[c][0]) is str:
            length = df[c].str.len().sum()
            lengths = np.append(lengths, length)
        else:
            lengths = np.append(lengths, 0)
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


def feature_extraction(file_name, model_name, n, batch_size=16, iterations=1):
    print(file_name)
    df = pd.read_csv(file_name, encoding = "ISO-8859-1")
    df = preprocess(df, n)
    data, data_attributes = df[df.columns[0]], df[df.columns[1:]]

    feature_dict = {}
    mapping = {}
    attributes = {}
    for j in range(iterations):
        for i, text in enumerate(data):
            feature_dict[i] = text
            mapping[i + len(data)*j] = text + model_name + "_" + str(j)
            attributes[mapping[i + len(data)*j]] = dict(zip(list(data_attributes.columns), [attribute[i] for attribute in data_attributes.to_numpy().T]))
            attributes[mapping[i + len(data)*j]]["Model Name"] = model_name
    model = SentenceTransformer(model_name, trust_remote_code=True)
    feature_list = list(feature_dict.values())
    for i in range(iterations):
        if i == 0:
            with torch.no_grad():
                feature_extract = batch_encode(model, feature_list, batch_size)
        if i>0:
            with torch.no_grad():
                feature_extract = np.vstack((feature_extract, batch_encode(model, feature_list, batch_size)))
    print(np.shape(feature_extract))
    return(feature_extract, mapping, attributes)

def gephi(feature_extract, file_name, model_name, mapping, attributes):
    nn = NN.kneighbors_graph(feature_extract, n_neighbors=4, mode="distance", metric="l1")

    _ = np.exp(-nn.data**2/np.mean(nn.data)**2, out=nn.data)

    G = nx.DiGraph(nn)
    H = nx.relabel_nodes(G, mapping)
    nx.set_node_attributes(H, attributes)
    nx.write_gexf(H, file_name + "_" +  model_name + ".gexf")

def gephi_export(feature_extract, file_name, model_name, mapping, attributes, n):
    file_name = file_name.replace(".csv", "")
    file_name = file_name.replace("data\\", "")
    model_name = model_name.replace("/", "_")
    gephi(feature_extract, file_name, model_name, mapping, attributes)

def run_all(datasets, models, n, graph=False, batch_size=16, iterations=1):
    model_dict = {}
    for model in models:
        print(model)
        for i, dataset in enumerate(datasets):
            if i == 0:
                feature_extract, mapping, attributes = feature_extraction(dataset, model, n, batch_size, iterations)
                if graph == True: gephi_export(feature_extract, dataset, model, mapping, attributes, n)
            else:
                feature_extract = np.dstack((feature_extract, feature_extraction(dataset, model, n, batch_size, iterations)[0]))
                if graph == True: gephi_export(feature_extract, dataset, model, mapping, attributes, n)
        model_dict[model] = feature_extract
    return(model_dict)
