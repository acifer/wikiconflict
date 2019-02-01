import sys,os
sys.path.append("../")
import pandas as pd
import numpy as np
import requests
import pickle
import traceback
from scripts.wiki import Wiki,Revision

def create_change_object(article_name, content_dir = "../data/content/", 
                            change_object_dir =  "../data/change objects/", epsilon_size=30, save=False):
    
    content_filepath = os.path.join(content_dir, article_name+".h5")
    change_object_filepath = os.path.join(change_object_dir, article_name+".pkl")
    
    with pd.HDFStore(content_filepath, 'r') as store:
        #retrieving all rev list and change object from file
        rev_list = store.get("rev_list")
        all_rev = store.get("all_tokens")
        all_tokens = all_rev.to_dict(orient="records")
        
        #making revision objects
        revs = rev_list.apply(lambda rev: Revision(rev["id"],rev["timestamp"], rev["editor"]),axis=1)
        revs.index = rev_list.id
        
        # Getting first revision object and adding content ot it
        from_rev_id = revs.index[0]
        wiki = Wiki(2345, article_name, revs, all_tokens)
        wiki.revisions.iloc[0].content = store["r"+str(from_rev_id)] 
        # adding content to all other revision and finding change object between them.
        
        for to_rev_id in list(revs.index[1:]):
            key="r"+str(to_rev_id)
            to_rev_content = store[key]
            wiki.create_change(from_rev_id, to_rev_id, to_rev_content, epsilon_size)
            from_rev_id = to_rev_id
         
    if save:
        with open(change_object_filepath, "wb") as file:
            pickle.dump(wiki, file)
        
    return wiki



if __name__ == "__main__":
    article_series=pd.read_csv("../conflicted_article.csv")["articles"]
    for article in article_series[19:]:
        print(article)
        
        len_file = article_name + "_rev_len.h5"
        len_file_path = os.path.join(content_dir, len_file)

        filename = article + ".h5"
        content_dir = "../data/content/"
        filepath = os.path.join(content_dir, filename)
        with pd.HDFStore(filepath, 'r') as store:
            #retrieving all rev list and change object from file
            rev_list = store.get("rev_list")["id"].values.tolist()
            keys = ["r" +  str(rev) for rev in rev_list]
            rev_len_list = [store.get(key).shape[0] for key in keys]
            rev_len_df = pd.DataFrame({"rev_id":rev_list[:-1], "length": rev_len_list[:-1]})
            rev_len_df.to_hdf(len_file_path, "rev_len")
        
        wiki = create_change_object(article_name, save=False)
        change_objects = []
        wiki.revisions.iloc[:-1].apply(lambda revision: change_objects.append(revision.change_df))
        timestamp_s = pd.to_datetime([ rev.timestamp for rev in    wiki.revisions.values.ravel().tolist()])
        time_gap = pd.to_timedelta(timestamp_s[1:]-timestamp_s[:-1])

        rev_ids = [ rev.id for rev in  wiki.revisions.tolist()]
        from_rev_ids = rev_ids[:-1]
        to_rev_ids= rev_ids[1:]

        editor_s = [ rev.editor for rev in  wiki.revisions.tolist()]

        index = list(zip(*[from_rev_ids, to_rev_ids, timestamp_s.tolist()[1:], time_gap, editor_s[1:]]))
        change_df = pd.concat(change_objects, sort=False, keys=index, names=["from revision id", "to revision id", "timestamp", "timegap", "editor"])
        change_object_dir =  "../data/change objects/"
        change_dataframe_path = os.path.join(change_object_dir, article_name+"_change.h5")
        change_df.to_hdf(change_dataframe_path, key="data")
