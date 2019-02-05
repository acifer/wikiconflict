
import requests
import numpy as np
import pandas as pd
import os
import traceback

def get_contents(baseurl, content, start_rev_id, end_rev_id=""):
    content_url = os.path.join(baseurl, "rev_content", content, str(start_rev_id)+"/")
    if end_rev_id:
        content_url = os.path.join(content_url, str(end_rev_id)+"/")
    params = { "o_rev_id": "false", "editor": "false", "token_id": "true", "in": "false", "out": "false" }
    try:
        response = requests.get(content_url, params= params)
        if response.status_code == requests.codes.ok: 
            response = response.json()
            if "revisions" in response.keys() :
                return response["revisions"]
            elif "revisions" not in response.keys() : 
                raise AttributeError("Server did not return revisions key it returned \t"+response.keys())
        elif response.status_code != requests.codes.ok : 
            raise AttributeError("Server returned bad code\t"+response.status_code)
    except:
        print(traceback.format_exc())

#pads each revisions content with a start and end 
def tokens_to_df(tokens):
    tokens.insert(0, {'token_id':-1, 'str':  "{st@rt}"})
    tokens.append({'token_id':-2, 'str': "{$nd}"})
    return pd.DataFrame(tokens)



def save_content(revison_series, filename, content, step=200, baseurl="https://api.wikiwho.net/en/api/v1.0.0-beta/"):
    end_index = revison_series.size
    from_index = 0
    with pd.HDFStore(filename, 'a') as store:
        try:
            for to_index in  range(0, end_index, step):    
                rev_contents = get_contents(baseurl, content, str(revison_series[from_index]), str(revison_series[to_index]))
                from_index = to_index
                for rev_content in rev_contents:
                    key = "r"+list(rev_content.keys())[0]
                    df = tokens_to_df(list(rev_content.values())[0]["tokens"])
                    store.put(key, df, table=False)
            to_index = from_index + (end_index-1)%step
            rev_contents = get_contents(baseurl, content, str(revison_series[from_index]), str(revison_series[to_index]))
            rev_contents.extend(get_contents(baseurl, content, str(revison_series[to_index])))
            for rev_content in rev_contents:
                key = "r"+list(rev_content.keys())[0]
                df = tokens_to_df(list(rev_content.values())[0]["tokens"])
                store.put(key, df, table=False)
        except:
            print("problem ", traceback.format_exc())


# In[5]:


def save_article(article_name, baseurl="https://api.wikiwho.net/en/api/v1.0.0-beta/", step=200):
    params = {"editor": "true", "timestamp": "true"}
    filename = article_name + ".h5"
    revisions_url = os.path.join( baseurl, "rev_ids", article_name+"/")
    response = requests.get(revisions_url, params= params)
    revisons_list = response.json()["revisions"]
    rev_list_df = pd.DataFrame(revisons_list)
    with pd.HDFStore(filename, 'a') as store:
        store.put("rev_list", rev_list_df, table=False)
    save_content(rev_list_df["id"], filename, article_name, step=step)




# In[8]:

if __name__ == "__main__":
    #article_series=pd.read_csv("conflicted_article.csv")["articles"]
    article_series = ["Hummus", "Andy Murray", "Istanbul"]
    print("starting download for the conflicted articles")
    for article in article_series:
        print("Downlaoding the article", article)    
        save_article(article)
    print("finishing download")

