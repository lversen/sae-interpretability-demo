from transformers import pipeline
import numpy as np
import sklearn.neighbors as NN
import networkx as nx
from sentence_transformers import SentenceTransformer


def remove_first_sentence(input):
    if type(input) is str:
        output = input.split(". ", maxsplit=1)
        output = output[1:]
        return (output)


def only_first_paragraph(input):
    if type(input) is str:
        output = input.split("\n", maxsplit=1)
        output = output[0]
        return (output)


def split(input):
    if type(input) is str:
        return (input.split(","))


def intersection(**kwargs):
    list1 = kwargs.get("list1", None)
    list2 = kwargs.get("list2", None)
    fd1 = kwargs.get("feature_dict_1", None)
    fd2 = kwargs.get("feature_dict_2", None)
    if fd1 and fd2 is not None:
        list1 = np.array(list(fd1.keys()))
        list2 = np.array(list(fd2.keys()))
    if len(list1) <= len(list2):
        intersection = np.intersect1d(list1, list2, return_indices=True)
        unsort = np.sort(intersection[1])
        if fd2 is not None:
            keyorder = list(list1[unsort].reshape(-1))
            fd2 = {k: fd2[k] for k in keyorder}
            fd1 = {k: fd1[k] for k in keyorder}
            return (fd1, fd2)
        return (list1[unsort])
    else:
        intersection = np.intersect1d(list2, list1, return_indices=True)
        unsort = np.sort(intersection[1])
        if fd1 and fd2 is not None:
            keyorder = list(list2[unsort].reshape(-1))
            fd1 = {k: fd1[k] for k in keyorder}
            fd2 = {k: fd2[k] for k in keyorder}
            return (fd1, fd2)
        return (list2[unsort])


def return_dictionaries(movie_list):
    import pandas as pd
    import numpy as np
    df = pd.read_csv('data/final_data.csv', sep=',')
    df = df.sort_values("RatingValue", ascending=True, ignore_index=True)
    df = df.drop(columns=["id", "DatePublished", "Keywords", "duration",
                          "Actors", "Director", "url", "PosterLink", "RatingCount",
                          "BestRating", "WorstRating", "ReviewAurthor",
                          "ReviewDate", "ReviewBody"])
    df = df.iloc[::-1]
    df = df.dropna()

    movies = movie_list
    all_movies = df["Name"].to_numpy()
    movies = intersection(list1=movies, list2=all_movies)

    genres = {}
    movie_dict = {}
    counter = 0
    for i in movies:
        i = i[0]
        """ genres[counter] = [
            split(df.loc[df["Name"] == i]["Genres"].to_numpy()[0])[0]] """
        genres[counter] = df.loc[df["Name"] == i]["Genres"].to_numpy()[0]
        """ genres[counter].append(
            len(split(df.loc[df["Name"] == i]["Genres"].to_numpy()[0]))) """
        print(genres[counter])
        movie_dict[counter] = [i]
        counter += 1
    return (movie_dict, genres)


def gephi_export(feature_dict, movie_dict, genres, file_name):
    # =============================================================================
    #     model = SentenceTransformer('sentence-transformers/paraphrase-TinyBERT-L6-v2')
    # =============================================================================
    model = SentenceTransformer('whaleloops/phrase-bert')
    feature_list = [text for features in feature_dict.values()
                    for text in np.unique(features)]
    feature_extract = model.encode(feature_list)

    nn = NN.kneighbors_graph(
        feature_extract, n_neighbors=4, mode="distance", metric="l1")

    _ = np.exp(-nn.data**2/np.mean(nn.data)**2, out=nn.data)
    print(_)
    G = nx.DiGraph(nn)

    mapping = {}
    attributes = {}
    counter = 0

    for attribute, values in movie_dict.items():
        for value in values:
            mapping[counter] = value
            counter += 1
            print(value)

    counter = 0
    for attribute, values in genres.items():
        print(values)
        attributes[mapping[counter]] = {
            "Genre": values} # , "Number of Genres": values[1]
        counter += 1

    H = nx.relabel_nodes(G, mapping)
    nx.set_node_attributes(H, attributes)
    nx.write_gexf(H, file_name)

def distance(dataset_1, dataset_2):
    feature_dict_1, model_1 = dataset_1
    feature_dict_2, model_2 = dataset_2
    fd1, fd2 = intersection(feature_dict_1=feature_dict_1,
                            feature_dict_2=feature_dict_2)
# =============================================================================
#     model = SentenceTransformer('whaleloops/phrase-bert')
# =============================================================================
    feature_list_1 = [text for features in fd1.values()
                      for text in np.unique(features)]
    feature_extract_1 = model_1.encode(feature_list_1)
    feature_list_2 = [text for features in fd2.values()
                      for text in np.unique(features)]
    feature_extract_2 = model_2.encode(feature_list_2)
    m1 = feature_extract_1
    m2 = feature_extract_2
    d = np.sqrt((m1-m2)**2)
    return (d)


def mean_distance(d):
    return (np.mean(d))


def std_distance(d):
    return (np.std(d))


def add_dataset(feature_dict, model):
    if "datasets" not in globals():
        global datasets
        datasets = np.array([feature_dict, model], dtype=object)
        return(datasets)
    else:
        datasets = np.c_[datasets, np.array(
            [feature_dict, model], dtype=object)]
        return(datasets)


def dist_matrix(datasets):
    datasets = datasets.T
    mean_matrix = np.empty((np.shape(datasets)[0], np.shape(datasets)[0]))
    std_matrix = np.empty((np.shape(datasets)[0], np.shape(datasets)[0]))
    for i, dataset_i in enumerate(datasets):
        for j, dataset_j in enumerate(datasets):
            mean_matrix[i][j] = mean_distance(distance(dataset_i, dataset_j))
            std_matrix[i][j] = std_distance(distance(dataset_i, dataset_j))
    return(np.array([mean_matrix, std_matrix]))