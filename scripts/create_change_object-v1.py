import sys,os
sys.path.append("../")
import pandas as pd
import numpy as np
import requests
import pickle
import traceback
from scripts.wiki import Wiki,Revision

def create_and_save_change_object(article_name, content_dir = "../data/content/", 
                            change_object_dir =  "../data/change objects/", epsilon_size=30):
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
        wiki = Wiki(2345, content, revs, all_tokens)
        wiki.revisions.iloc[0].content = store["r"+str(from_rev_id)] 
        # adding content to all other revision and finding change object between them.
        for to_rev_id in list(revs.index[1:]):
            key="r"+str(to_rev_id)
            to_rev_content = store[key]
            wiki.create_change(from_rev_id, to_rev_id, to_rev_content, epsilon_size)
            from_rev_id = to_rev_id
    with open(change_object_filepath, "wb") as file:
        pickle.dump(wiki, file)

if __name__ == "__main__":
    article_series=pd.read_csv("../conflicted_article.csv")["articles"]
    for article in article_series[19:]:
        print(article)
        create_and_save_change_object(article)