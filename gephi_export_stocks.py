import sklearn.neighbors as NN
import networkx as nx
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
import sys
np.set_printoptions(threshold=sys.maxsize, suppress=False)

def language_classifier(df, rows, max_rows, columns, file_name):
    model_name = 'qanastek/51-languages-classifier'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    classifier = TextClassificationPipeline(model=model, tokenizer=tokenizer, device="cuda")

    df_max = pd.read_csv(file_name, encoding = "ISO-8859-1")

    for c in columns:
        c_classified = c + "_classified"
        if c_classified not in df.columns:    
            if c in df.columns:
                print("Classifying languages from '" + c + "'")
                data = classifier(list(df[c].to_numpy()))
                data = [d["label"] for d in data]
                df[c_classified] = data

                data_max = np.zeros(max_rows, dtype='U100')
                data_max[np.arange(max_rows)] = "empty"
                data_max[rows] = data
                df_max[c_classified] = data_max
                df_max.to_csv("data_movies\\final_data.csv", index=False)
            else: raise ValueError(c + " is not a column in " + file_name)
        else:
            rows_remaining = np.array([])
            data = np.array([], dtype='U100')
            for r in range(len(df)):
                d_point = df.loc[r][c_classified]
                if d_point == "empty":
                    rows_remaining = np.append(rows_remaining, r)
                    data = np.append(data, classifier(df.loc[r][c]))
                else:
                    data = np.append(data, d_point)
            print(rows_remaining)
            
            


def batch_encode(model, feature_list, batch_size, n):
    embeddings = []
    for i in range(0, len(feature_list), batch_size):
        batch = feature_list[i:i+batch_size]
        batch_embeddings = model.encode(batch, device='cuda')
        embeddings.extend(batch_embeddings)
        if i>0:
            if int(i*100/n) > int((i-1)*100/n):
                print(str(int(i*100/n)) + "%" + " encoding")
        torch.cuda.empty_cache()
    return np.array(embeddings)

def preprocess(df, file_name, content_column, dataset_iteration, n):
    if "Unnamed: 0" in df.columns:
        l = list(df.columns)
        l.remove("Unnamed: 0")
        df = df[l]
    if "id" in df.columns:
        l = list(df.columns)
        l.remove("id")
        df = df[l]
    if len(content_column) != 0:
        data_name = content_column[dataset_iteration]
        if data_name in df.columns:
            l = list(df.columns)
            l.remove(data_name)
            l = np.append([data_name], l)
            df = df[l]
            rows = np.random.choice(df.index, size=n, replace=False)
            max_rows = len(df)
            df = df.iloc[rows]
            df = df.reset_index()
            df = df.drop(columns="index")
            df[df.columns[0]] = df[df.columns[0]].str.strip()
            return(df, rows, max_rows)
        else: raise ValueError(data_name + " is not a column in " + file_name)
    else:
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
        max_rows = len(df)
        df = df.iloc[rows]
        df = df.reset_index()
        df = df.drop(columns="index")
        df[df.columns[0]] = df[df.columns[0]].str.strip()
        return(df, rows, max_rows)

def data_frame_init(file_name, content_column, dataset_iteration, n):
    df = pd.read_csv(file_name, encoding = "ISO-8859-1")
    df, rows, max_rows = preprocess(df, file_name, content_column,  dataset_iteration, n)
    return(df, rows, max_rows)

def node_attributes(df, file_name, dataset_iteration, title_column):
    if title_column[dataset_iteration] in df.columns:
        data, data_attributes = df[df.columns[0]], df[df.columns[1:]]
        mapping = dict(zip(np.arange(len(df)), df[title_column[dataset_iteration]]))
        attributes = dict(zip(df[title_column[dataset_iteration]], [dict(zip(df.columns, df.loc[i])) for i in np.arange(len(df))]))
        return(mapping, attributes)
    else: raise ValueError(title_column[dataset_iteration] + " is not a column in " + file_name)

def feature_extraction(df, model_name, batch_size, n):
    feature_list = df[df.columns[0]].to_numpy()
    model = SentenceTransformer(model_name, trust_remote_code=True)
    with torch.no_grad():
        feature_extract = batch_encode(model, feature_list, batch_size, n)
    print(np.shape(feature_extract))
    return(feature_extract)

def gephi(feature_extract, file_name, model_name, mapping, attributes):
    nn = NN.kneighbors_graph(feature_extract, n_neighbors=4, mode="distance", metric="l1")

    _ = np.exp(-nn.data**2/np.mean(nn.data)**2, out=nn.data)

    G = nx.DiGraph(nn)
    H = nx.relabel_nodes(G, mapping, copy=False)
    nx.set_node_attributes(H, attributes)
    nx.write_gexf(H, file_name + "_" +  model_name + ".gexf")

def gephi_export(feature_extract, file_name, model_name, mapping, attributes):
    file_name = file_name.replace(".csv", "")
    file_name = file_name.replace("data\\", "")
    model_name = model_name.replace("/", "_")
    gephi(feature_extract, file_name, model_name, mapping, attributes)

def run_all(datasets, models, n, graph=False, batch_size=16, content_column=[], title_column=[], classify_language=[]):
    model_dict = {}
    for i, dataset in enumerate(datasets):
        print(dataset)
        df, rows, max_rows = data_frame_init(dataset, content_column, i, n)
        if len(classify_language) != 0:
            language_classifier(df, rows, max_rows, classify_language, dataset)
            mapping, attributes = node_attributes(df, dataset, i, title_column)

        for j, model in enumerate(models):
            feature_extract = feature_extraction(df, model, batch_size, n)
            if graph == True: gephi_export(feature_extract, dataset, model, mapping, attributes)
        
        if len(title_column) != 0:
            title_column = np.delete(title_column, i)
        if len(content_column) != 0:
            content_column = np.delete(content_column, i)
    return(model_dict)
"""  else:
                feature_extract = np.dstack((feature_extract, feature_extraction(df, dataset, model, i, batch_size, iterations, title_column)[0]))
                print(feature_extract.shape)
                if graph == True: gephi_export(feature_extract, dataset, model, mapping, attributes, n)
 """
            #model_dict[model] = feature_extract
