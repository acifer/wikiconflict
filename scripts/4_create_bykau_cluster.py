"""

"""

import sys,os
sys.path.append("../")

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from IPython.display import HTML
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
from scipy import stats
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MultiLabelBinarizer


from sklearn.metrics import adjusted_mutual_info_score
from sklearn.metrics import pairwise_distances  
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import mutual_info_score
from sklearn.metrics import normalized_mutual_info_score

article_name = "John_Logie_Baird"
change_object_dir =  "../data/change objects/"
content_dir = "../data/content/"
filename =  f"{article_name}_change.h5"
change_object_file = os.path.join(change_object_dir, filename)
filename = article_name + ".h5"
filepath = os.path.join(content_dir, filename)

def read_all_token(article_name, content_dir="../data/content/",):
    filename = article_name + ".h5"
    filepath = os.path.join(content_dir, filename)

    if os.path.exists(filepath.encode("utf-8")):
        with pd.HDFStore(filepath, 'r') as store:
             token_string_df = store.get("all_tokens")
        token_string_df = token_string_df.set_index("token_id")["str"]
        token_string_df[-1] = "St@rt"
        token_string_df[-2] = "$nd"
        return token_string_df

    else:
        print(f"{filepath} not found")
        raise(FileNotFoundError)


def read_change_object(article_name, change_object_dir =  "../data/change objects/" ):
    filename =  f"{article_name}_change.h5"
    change_object_file = os.path.join(change_object_dir, filename)
    
    if os.path.exists(change_object_file.encode("utf-8")):
        with pd.HDFStore(change_object_file, 'r') as store:
            change_object_dataframe = store.get("data")
        return change_object_dataframe
    else:
        print(f"{change_object_file} not found")
        raise(FileNotFoundError)



def add_length(change_object_dataframe):
    change_object_dataframe["ins_length"] = change_object_dataframe["ins_tokens"].str.len()
    change_object_dataframe["del_length"] = change_object_dataframe["del_tokens"].str.len()
    return change_object_dataframe

def convert_wiki_who_to_string(change_object_dataframe, token_string_df):
    change_object_dataframe["del_string_tokens"] = change_object_dataframe["del_tokens"].apply(
        lambda x:  tuple(token_string_df[np.array(x)].tolist()))

    change_object_dataframe["ins_string_tokens"] = change_object_dataframe["ins_tokens"].apply(
        lambda x:  tuple(token_string_df[np.array(x)].tolist()))

    change_object_dataframe["left_context"] = change_object_dataframe["left_token"].apply(
        lambda x:  tuple(token_string_df[np.array(x)].tolist())).str.join(" ")


    change_object_dataframe["right_context"] = change_object_dataframe["right_token"].apply(
        lambda x:  tuple(token_string_df[np.array(x)].tolist())).str.join(" ")
    return change_object_dataframe




def init_bykau_cluster(change_object_dataframe):
    change_object_dataframe["bykau_cluster"] = pd.Series(-99,index=change_object_dataframe.index)
    return change_object_dataframe

def get_ins_and_del(change_object_dataframe):
    ins_and_del = change_object_dataframe[(change_object_dataframe["ins_string_tokens"]!=()) & (change_object_dataframe["del_string_tokens"]!=())]
    return ins_and_del

def filter_on_size(ins_and_del, max_gap_size =5):
    reduced_ins_and_del = ins_and_del[~((ins_and_del["ins_length"] > max_gap_size ) | (ins_and_del["del_length"] > max_gap_size) )]
    return reduced_ins_and_del

def filter_on_user_support(reduced_ins_and_del, min_no_of_user=2):
    bykau_change_object = reduced_ins_and_del.groupby("ins_string_tokens").filter(lambda x : x.index.get_level_values("editor").nunique()>= min_no_of_user)
    bykau_change_object = bykau_change_object.groupby("del_string_tokens").filter(lambda x : x.index.get_level_values("editor").nunique() >= min_no_of_user)
    return bykau_change_object

def cluster(bykau_change_object, r_threshold = 8, cutoff_threshold = 0.75):

    left_neighbours = bykau_change_object["left_context"].apply(lambda x: x.split(" ")[-r_threshold:])
    right_neighbours = bykau_change_object["right_context"].apply(lambda x: x.split(" ")[:r_threshold])
    neighbour_tokens = left_neighbours + right_neighbours
    
    neighbour_tokens_set = neighbour_tokens.apply(lambda x: np.unique(x))
    neighbour_vec = MultiLabelBinarizer().fit_transform(neighbour_tokens_set)

    db = DBSCAN(eps=cutoff_threshold, min_samples=5, metric='jaccard').fit(neighbour_vec)
    return db.labels_


def save_clusters(article_name, change_object_dataframe, bykau_dir= "../data/bykau_change_object/"):
    filename =  f"{article_name}_change.h5"

    change_object_file = os.path.join("../data/bykau_change_object/", filename)
    with pd.HDFStore(change_object_file, 'w') as store:
        store.put("data", change_object_dataframe["bykau_cluster"],)  
        
def main(article_name="John_Logie_Baird", content_dir="../data/content/",
        bykau_dir =  "../data/bykau_change_object/", change_object_dir =  "../data/change objects/",
        max_gap_size =5, min_no_of_user=2, r_threshold =8, cutoff_threshold =0.75):

    token_str_map = read_all_token(article_name)
    change_object_dataframe = read_change_object(article_name)
    change_object_dataframe = add_length(change_object_dataframe)
    change_object_dataframe = convert_wiki_who_to_string(change_object_dataframe, token_str_map)
    change_object_dataframe = init_bykau_cluster(change_object_dataframe)
    
    ins_and_del_dataframe = get_ins_and_del(change_object_dataframe)
    reduced_on_size_dataframe = filter_on_size(ins_and_del_dataframe, max_gap_size=max_gap_size)
    bykau_change_object = filter_on_user_support(reduced_on_size_dataframe, min_no_of_user=min_no_of_user)
    
    clusters = cluster(bykau_change_object, r_threshold =r_threshold, cutoff_threshold=cutoff_threshold)
    change_object_dataframe.loc[bykau_change_object.index,"bykau_cluster"]= clusters
    save_clusters(article_name, change_object_dataframe)

if __name__ == "__main__":
    list_of_articles=pd.read_csv("../conflicted_article.csv")["articles"].tolist()
    for article in list_of_articles:
        print(f"processing article name {article}")
        main(article_name=article)
    
    big_list_of_articles = pd.read_csv("../conflicted_article-big.csv")["articles"].tolist()

    for article in big_list_of_articles:
        print(f"processing article name {article}")
        main(article_name=article)
        
