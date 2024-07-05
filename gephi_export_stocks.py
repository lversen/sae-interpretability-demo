import sklearn.neighbors as NN
import networkx as nx
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
import torch


from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
def classify_language(text):
    model_name = 'qanastek/51-languages-classifier'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    classifier = TextClassificationPipeline(model=model, tokenizer=tokenizer)
    res = classifier(text)[0]["label"]
    return(res)


def batch_encode(model, feature_list, batch_size=16):
    embeddings = []
    for i in range(0, len(feature_list), batch_size):
        batch = feature_list[i:i+batch_size]
        batch_embeddings = model.encode(batch, device='cuda')  # Specify device
        embeddings.extend(batch_embeddings)
        print(str(i) + " encoding")
        torch.cuda.empty_cache()  # Clear cache after each batch
    return np.array(embeddings)

def preprocess(df, content_column, dataset_iteration, n):
    if "id" in df.columns:
        l = list(df.columns)
        l.remove("id")
        df = df[l]
    if len(content_column) != 0:
        data_name = content_column[dataset_iteration]
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
        df = df.iloc[rows]
        df = df.reset_index()
        df = df.drop(columns="index")
        df[df.columns[0]] = df[df.columns[0]].str.strip()
        return(df)

def data_frame_init(file_name, content_column, dataset_iteration, n):
    df = pd.read_csv(file_name, encoding = "ISO-8859-1")
    df = preprocess(df, content_column,  dataset_iteration, n)
    return(df)

def feature_extraction(df, file_name, model_name, dataset_iteration, batch_size, iterations, title_column, language_classes):
    title_column = np.array(title_column)
    data, data_attributes = df[df.columns[0]], df[df.columns[1:]]
    feature_dict = {}
    mapping = {}
    attributes = {}
    for j in range(iterations):
        for i, text in enumerate(data):
            try:
                feature_dict[i] = text
                language = classify_language(text)
            except ValueError:
                continue
            print(str(i + len(data)*j ) + " features, mapping and attributes")
            if len(title_column) == 0:
                mapping[i + len(data)*j] = str(i + len(data)*j) + file_name + "_" + model_name + "_" + str(j)
            else:
                mapping[i + len(data)*j] = df[title_column[dataset_iteration]][i + len(data)*j] + "\\" + model_name
            attributes[mapping[i + len(data)*j]] = dict(zip(list(data_attributes.columns), [attribute[i] for attribute in data_attributes.to_numpy().T]))
            attributes[mapping[i + len(data)*j]]["Model Name"] = model_name
            if language_classes is True:
                attributes[mapping[i + len(data)*j]]["Language"] = language
                print(language)

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
    H = nx.relabel_nodes(G, mapping, copy=False)
    nx.set_node_attributes(H, attributes)
    nx.write_gexf(H, file_name + "_" +  model_name + ".gexf")

def gephi_export(feature_extract, file_name, model_name, mapping, attributes):
    file_name = file_name.replace(".csv", "")
    file_name = file_name.replace("data\\", "")
    model_name = model_name.replace("/", "_")
    gephi(feature_extract, file_name, model_name, mapping, attributes)

def run_all(datasets, models, n, graph=False, batch_size=16, iterations=1, content_column=[], title_column=[], language_classes=False):
    model_dict = {}
    for i, dataset in enumerate(datasets):
        print(dataset)
        df = data_frame_init(dataset, content_column, i, n)

        for j, model in enumerate(models):
            print(model)
            feature_extract, mapping, attributes = feature_extraction(df, dataset, model, i, batch_size, iterations, title_column, language_classes)
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
