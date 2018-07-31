import pandas as pd
import numpy as np
import traceback

class Wiki:
    def __init__(self,id,title, revs, all_tokens=[]):
        self.id = id
        self.title = title
        self.revisions = revs
        self.add_all_token(all_tokens)
        
    def add_all_token(self, all_tokens):
        for token in all_tokens:
            self.revisions.loc[token["o_rev_id"]].added.add(token["token_id"])
            for in_revision in token["in"]:
                self.revisions.loc[in_revision].added.add(token["token_id"])
            for out_revision in token["out"]:
                self.revisions.loc[out_revision].removed.add(token["token_id"])
                
    def create_change(self, from_rev_id, to_rev_id, to_rev_content, vocab, epsilon_size):
        try:
            from_rev = self.revisions[from_rev_id]
            to_rev = self.revisions[to_rev_id]
            from_rev.deleted(to_rev)
            from_rev.content["invocab"] = from_rev.content["str"].isin(vocab)
            to_rev.content = to_rev_content
            to_rev.inserted_continuous_pos()
            to_rev.inserted_neighbours()
            from_rev.create_change_object(to_rev)
            from_rev.append_neighbour_vec(to_rev, epsilon_size)
        except:
            print("exception occurred in calculating change object",traceback.format_exc())
            print("problem in ", rev_content.keys() )


class Revision:
    def __init__(self, id, timestamp,editor):
        self.id = id
        self.timestamp = timestamp
        self.editor = editor
        self.added = set()
        self.removed = set()   
        
    def deleted(self, to_rev):
        self.content["removed"] = pd.Series(np.isin( self.content["token_id"].values, list(to_rev.removed), assume_unique= True ))
        end_pos = np.argwhere(np.ediff1d(np.pad(self.content["removed"].astype(np.int), (1,1), mode="constant", constant_values=0)) == -1) -1 
        start_pos = np.argwhere(np.ediff1d(np.pad(self.content["removed"].astype(np.int), (1,1), mode="constant", constant_values=0)) == 1)
        start_neighbour = start_pos - 1
        end_neighbour = end_pos + 1
        self.deleted_object = pd.DataFrame(np.c_[ start_pos, end_pos, start_neighbour, end_neighbour ],
                                       columns=[ "del_start_pos", "del_end_pos", "left_neigh", "right_neigh",])
    
    def inserted_continuous_pos(self):
        self.content["added"] = pd.Series(np.isin( self.content["token_id"].values, list(self.added), assume_unique= True))
        end_pos = np.argwhere(np.ediff1d(np.pad(self.content["added"].astype(np.int), (1,1), mode="constant", constant_values=0)) == -1) -1 
        start_pos = np.argwhere(np.ediff1d(np.pad(self.content["added"].astype(np.int), (1,1), mode="constant", constant_values=0)) == 1)
        self.added_pos = np.c_[start_pos, end_pos]

    def inserted_neighbours(self):
        start_token_pos = self.added_pos[:,0] - 1
        end_token_pos = self.added_pos[:,1] + 1
        self.start_token_id = self.content["token_id"].values[start_token_pos]
        self.end_token_id = self.content["token_id"].values[end_token_pos]
    
    def create_change_object(self, to_rev):
        self.ins_left = np.argwhere(np.isin(self.content.token_id.values, to_rev.start_token_id, assume_unique= True))
        self.ins_right = np.argwhere(np.isin(self.content.token_id.values, to_rev.end_token_id, assume_unique= True))
        self.inserted_object = pd.DataFrame(np.concatenate([to_rev.added_pos, self.ins_left, self.ins_right], axis=1),
                                       columns=["ins_start_pos", "ins_end_pos", "left_neigh", "right_neigh", ])

        self.change = pd.merge(self.inserted_object, self.deleted_object,how="outer", on=["left_neigh", "right_neigh"])
        self.change.fillna(-1, inplace=True)
        
    def append_neighbour_vec(self, to_rev, epsilon_size):
        self.vocabs_pos = np.argwhere( self.content["invocab"].values)
        self.content_str_vec = self.content.str.values
        del self.content
        neighbour_df = self.change.apply(find_tokens, axis=1, args=(self,to_rev, epsilon_size))
        neighbour_df.columns= ["ins_tokens", "del_tokens", "left_neigh", "right_neigh", "left_token", "right_token"]
        self.neighbour = neighbour_df
        self.change_df = pd.concat([self.change, neighbour_df], sort=False, axis=1)
        


def find_tokens(change, revision, to_rev, epsilon_size):
    left_neigh = revision.vocabs_pos[revision.vocabs_pos <= change["left_neigh"]][-epsilon_size:]
    right_neigh = revision.vocabs_pos[revision.vocabs_pos >= change["right_neigh"]][:epsilon_size]
    if(change["ins_start_pos"]==-1):
        ins_tokens = []
    else:
        ins_slice = slice(int(change["ins_start_pos"]), int(change["ins_end_pos"]+1) )
        ins_tokens = to_rev.content.str.values[ins_slice]
    if(change["del_start_pos"] == -1):
        del_tokens = []
    else:
        del_slice = slice(int(change["del_start_pos"]), int(change["del_end_pos"]+1) )
        del_tokens = revision.content_str_vec[del_slice]
    left_token = revision.content_str_vec[left_neigh]
    right_token = revision.content_str_vec[right_neigh]
    return pd.Series([tuple(ins_tokens), tuple(del_tokens), tuple(left_neigh), tuple(right_neigh), tuple(left_token), tuple(right_token)])
