
from gensim.models import KeyedVectors
from gensim.models.fasttext import FastText as FT_gensim
import pandas as pd
import numpy as np
import requests
import pickle
import os
import traceback
from wiki import *


def get_word_vecs(wiki_vec, tokens):
    in_vocab_tokens = set(tokens) & set(wiki_vec.vocab)
    if in_vocab_tokens:
        return wiki_vec[in_vocab_tokens].sum(axis=0, keepdims=True)
    else:
        return np.zeros((1, wiki_vec.vector_size))

if __name__ == "__main__":
    wiki_vec = KeyedVectors.load_word2vec_format('../wordvectors/wiki.en.vec', binary=False, limit=5000)
    vocab = set(wiki_vec.vocab)
    baseurl = "https://api.wikiwho.net/en/api/v1.0.0-beta/"
    content = "Cloning"
    filename = content + ".h5"
    epsilon_size = 6

    all_content_url = os.path.join(baseurl, "all_content", content +"/")
    params = { "o_rev_id": "true", "editor": "false", "token_id": "true", "in": "true", "out": "true" }
    all_rev_data = requests.get(all_content_url, params= params)
    all_tokens = all_rev_data.json()["all_tokens"]

    with pd.HDFStore(filename, 'r') as store:
        rev_list = store.get("rev_list")    
        revs = rev_list.apply(lambda rev: Revision(rev["id"],rev["timestamp"], rev["editor"]),axis=1)
        revs.index = rev_list.id
        from_rev_id = revs.index[0]
        wiki = Wiki(2345, content, revs, all_tokens)
        wiki.revisions.iloc[0].content = store["r"+str(from_rev_id)]   
        for to_rev_id in list(revs.index[1:]):
            key="r"+str(to_rev_id)
            to_rev_content = store[key]
            wiki.create_change(from_rev_id, to_rev_id, to_rev_content, vocab, epsilon_size)
            from_rev_id = to_rev_id
            


        with open(content+".pkl", "wb") as file:
            pickle.dump(wiki, file)
