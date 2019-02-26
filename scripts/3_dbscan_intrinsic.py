import sys,os
sys.path.append("../")

import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import pairwise_distances  

from scipy.stats import entropy
from sklearn.cluster import DBSCAN
from sklearn.metrics import pairwise_distances 
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import radius_neighbors_graph

def read_all_token(article_name, content_dir="../data/content/",):
    filename = article_name + ".h5"
    filepath = os.path.join(content_dir, filename)

    if os.path.exists(filepath):
        with pd.HDFStore(filepath, 'r') as store:
             token_string_df = store.get("all_tokens")
        token_string_df = token_string_df.set_index("token_id")["str"]
        token_string_df[-1] = "St@rt"
        token_string_df[-2] = "$nd"
        return token_string_df

    else:
        print(f"{filepath} file not found")
        raise(FileNotFoundError)
        return None


def read_change_object(article_name, change_object_dir =  "../data/change objects/" ):
    filename =  f"{article_name}_change.h5"
    change_object_file = os.path.join(change_object_dir, filename)
    
    if os.path.exists(change_object_file):
        with pd.HDFStore(change_object_file, 'r') as store:
            change_object_dataframe = store.get("data")
        return change_object_dataframe
    else:
        print(f"{change_object_file} file do not exist")
        raise(FileNotFoundError)

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


        


def read_revision_len(article_name):
    content_dir = "../data/content/"
    len_file = article_name + "_rev_len.h5"
    len_file_path = os.path.join(content_dir, len_file)
    all_revisions_length =  pd.read_hdf(len_file_path, key = "rev_len")
    return all_revisions_length

def calculate_relative_position(change_object_dataframe, all_revisions_length):
    change_object_dataframe = change_object_dataframe.reset_index().set_index('from revision id')
    change_object_dataframe = change_object_dataframe.join(all_revisions_length.set_index("rev_id"))
    change_object_dataframe.index.name = "from revision id"

    change_object_dataframe["relative_position"] =(change_object_dataframe["left_neigh"]+1)/(change_object_dataframe["length"]).round(3)

    change_object_dataframe = change_object_dataframe.reset_index().set_index(["from revision id","timestamp", "level_5"])

    return change_object_dataframe

def get_edit_tokens(change_object_dataframe):
    change_object_dataframe["edit_tokens"] = change_object_dataframe["ins_string_tokens"] + change_object_dataframe["del_string_tokens"]
    return change_object_dataframe


def gini(array):
    # Number of array elements:
    n = array.shape[0]
    index = np.arange(1, n+1)
    # Gini coefficient:
    return ((np.sum((2 * index - n  - 1) * array)) / (n * np.sum(array)))

def evaluate(change_object_dataframe, clusters, evaluation_df, column_names):
    
    change_object_dataframe["cluster"] = pd.Series(clusters, index= change_object_dataframe.index)
    
    stats_series = pd.Series(index=column_names)
    
    non_negative_cluster_mask = clusters != -1
    non_neg_cluster_df = change_object_dataframe.loc[non_negative_cluster_mask, :]

    stats_series["no_of_outliers"]  = np.count_nonzero(~non_negative_cluster_mask)
    stats_series["no_of_clusters"] = np.unique(clusters[non_negative_cluster_mask]).size

    rank_by_size = non_neg_cluster_df.groupby("cluster").size().sort_values(ascending=True)

    relative_postion_std = non_neg_cluster_df.groupby("cluster")["relative_position"].std()
#     stats_series["relative_position_std_max"] = relative_postion_std.max()
#     stats_series["relative_position_std_min"] = relative_postion_std.min()
#     stats_series["relative_position_std_median"] = relative_postion_std.median()
#     stats_series["relative_position_std_skewness"] = relative_postion_std.skew()
#     stats_series["relative_position_std_kurtosis"] = relative_postion_std.kurt()

    stats_series["relative_position_std_less_than_.1"] = np.count_nonzero(relative_postion_std <.1)
    

    size_stats = rank_by_size.describe()
    if rank_by_size.shape[0] > 1:
        stats_series["top2_ratio"] = rank_by_size.iloc[1]/rank_by_size.iloc[0]
    else:
        stats_series["top2_ratio"] = 0
    stats_series["max_cluster_size"] = size_stats["max"]
    stats_series["min_cluster_size"] = size_stats["min"]
    stats_series["mean_cluster_size"] = size_stats["mean"]
    stats_series["median_cluster_size"] = size_stats["50%"]
    stats_series["inter_quartile_range_cluster_size"] = size_stats["75%"] - size_stats["25%"]

    stats_series["variance_cluster_size"] = rank_by_size.var()
    stats_series["standard_deviation_cluster_size"] = size_stats["std"]
    stats_series["skewness_cluster_size"] = rank_by_size.skew()
    stats_series["kurtosis_cluster_size"] = rank_by_size.kurt()
    
    cluster_sizes = non_neg_cluster_df.groupby("cluster").size().values
    cluster_sizes = cluster_sizes / cluster_sizes.sum()
    
    token_entropy_clusters = non_neg_cluster_df.groupby("cluster")["edit_tokens"].apply(
        lambda token_tuples: entropy(pd.Series(
            [token for token_tuple in token_tuples.tolist() for token in token_tuple]
        ).value_counts().values))
    stats_series["token_entropy"] = (cluster_sizes * token_entropy_clusters).sum()
    stats_series["avg_token_entropy"] =  token_entropy_clusters.mean()
    
    position_entropy_clusters = non_neg_cluster_df.groupby("cluster")["relative_position"].apply(
        lambda x: entropy(x.value_counts().values))
    stats_series["position_entropy"] = (cluster_sizes * position_entropy_clusters).sum()
    stats_series["avg_position_entropy"] =  position_entropy_clusters.mean()

    stats_series["gini"] = gini(rank_by_size.values)
    
    change_object_dataframe  = change_object_dataframe.drop("cluster",axis=1)
    
    return stats_series


def read_change_vectors(article_name, change_vector_dir = "../data/change_vector/"):
    change_vec_filename = f"{article_name}.npz"
    change_vector_file = os.path.join(change_vector_dir, change_vec_filename)
    vectors ={}
    with open(change_vector_file, "rb") as file:
        arrays_dict = np.load(file)
        vectors["clean_notweighted_4"] = arrays_dict["4_clean_not_weighted"]
        vectors["clean_notweighted_10"] = arrays_dict["10_clean_not_weighted"] 
    return vectors

def cluster(distances, **dbscan_param):
    clusters = DBSCAN(**dbscan_param, metric="precomputed").fit(distances)
    return clusters.labels_

def cluster_and_evaluate(change_object_dataframe, vectors, vector_names, dbscan_params, 
                         evaluation_df, column_names):
    for cluster_by in vector_names:
        if change_object_dataframe.shape[0] <= 21173:
            distances = pairwise_distances(vectors[cluster_by])
        else:
            distances = radius_neighbors_graph( vectors[cluster_by], 4.02, mode="distance")
        for dbscan_param in dbscan_params:
            cluster_vec = cluster(distances, **dbscan_param)
            evaluation_df.loc[(cluster_by,dbscan_param["eps"]),:] = evaluate(
            change_object_dataframe, cluster_vec, evaluation_df, column_names)
    return evaluation_df

def save(evaluate_df, article_name, pre_evaluation_dir = "../data/pre_evaluation/"):
    evaluate_df["context_size"] =  pd.Series(evaluate_df.reset_index()[
        "types"].str.split("_", expand = True)[2].values, index=evaluate_df.index)
    file_name = f"{article_name}.csv"
    full_file_path = os.path.join(pre_evaluation_dir, file_name)
    evaluate_df.to_csv(full_file_path)
    return evaluate_df

def main(article_name="John_Logie_Baird", content_dir="../data/content/",
         change_object_dir =  "../data/change objects/",):
    token_str_map = read_all_token(article_name)
    change_object_dataframe = read_change_object(article_name)
    change_object_dataframe = convert_wiki_who_to_string(change_object_dataframe, token_str_map)
    all_revisions_length = read_revision_len(article_name)
    change_object_dataframe = calculate_relative_position(change_object_dataframe, all_revisions_length)
    change_object_dataframe = get_edit_tokens(change_object_dataframe)
    
    vector_names = [ "clean_notweighted_4", "clean_notweighted_10"]
    column_names = ["top2_ratio","no_of_outliers", "no_of_clusters",
                "relative_position_std_less_than_.1",
                "max_cluster_size", "min_cluster_size", "mean_cluster_size",
                "skewness_cluster_size", "kurtosis_cluster_size",
                 "median_cluster_size",  "inter_quartile_range_cluster_size",
                "variance_cluster_size", "standard_deviation_cluster_size",
                "gini", "token_entropy", "position_entropy",
               "avg_position_entropy", "avg_token_entropy"]
    
    dbscan_params = [{"eps": i, "min_samples":5} for i  in np.arange(0.25, 4.01, 0.25)]
    idx = pd.MultiIndex.from_product([vector_names, 
                                  [ param["eps"] for param in dbscan_params]],
                             names=['types', 'eps'])
    evaluation_df = pd.DataFrame(index=idx, columns=column_names)
    vectors = read_change_vectors(article_name)
    evaluate_df = cluster_and_evaluate(change_object_dataframe, vectors, vector_names, 
                                       dbscan_params, evaluation_df, column_names)
    evaluate_df = save(evaluate_df, article_name)
    return evaluate_df




if __name__ == "__main__":
#     list_of_articles=pd.read_csv("../conflicted_article.csv")["articles"].tolist()
#     for article in list_of_articles:
#         print(f"processing article name {article}")
#         main(article_name=article)
        
    list_of_articles=pd.read_csv("../conflicted_article-big.csv")["articles"].tolist()
    for article in list_of_articles:
        print(f"processing article name {article}")
        main(article_name=article)
#     res = main()
    print("finsihed")