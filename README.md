# wikiconflict

This repository is part of the analysis done for my master thesis. All the analysis steps are relased as IPython notebook.

Following steps needed to be performed for processing of the data.

## Download the Pre-Trained word vectors

Use fast text [word vectors](https://github.com/facebookresearch/fastText/blob/master/docs/pretrained-vectors.md) from [https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.en.vec]

## Creating required directory for storing the intermediate and final outcome of analysis.

Inside wikiconflict folder create the root folder /data for storing all the results of analysis.
#### Inside /data create various subfolders. Each of these subfolders will store data at various stage of processing. 
1. /content
2. /change objects
3. /change_vector
4. /bykau_change_object
5. /annotation

## Steps of analysis

### Downalod the Article
Use the notebook 1_download_rev_content.ipynb to download all the articles you wish to analyse.

### Ceate Change Object

Create change object with gaps of inserted and deleted tokens using the notebook 2_create_change_object-v2.ipynb

### Create Change Vector

Change Vectors created from pre-trained word vectors are one of the important paramter of the model. Use the notbook in 3_create_change_vector-v3.ipynb to create different change vectors corresponding to different values of context_length. All of these change vectors are saved in /change_vector folder. These change vectors will be used to create groups and analyse them.

### Cluster and Evaluate
Clustering is divided into intrinsic and extrinsic clustering 

### Compare with Bykau. et. Al.


