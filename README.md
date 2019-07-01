# wikiconflict

This repository is part of the analysis done for my master thesis.

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/acifer/wikiconflict/master?filepath=notebooks%2F4_2_clustering-example.ipynb)

## Research Goal
 Thesis aims to identify the group of edited tokens in the revision history of an article which are fine grained in their respective revisions.
 
In order to achieve our research goal, we break it into following steps.

1. We define and identify fine grained edit tokens, we call it Change Objects.
2. We transform Change Objects into Change Vectors of fixed dimension using pre-trained word vectors.
3. We create groups of Change Objects by clustering Change Vector.
4. We evaluate groups of fine grained Change Objects.
5. We compare our algorithm of identifying  group of Change Objects with algorithm proposed by [bykau et al].

All the analysis steps are released as IPython notebook. Further we also release example of cluster groups created on [John Logie Baird](https://en.wikipedia.org/wiki/John_Logie_Baird) in the [example notebook](https://github.com/acifer/wikiconflict/blob/master/notebooks/4_2_clustering-example.ipynb)



## Preparing the code.
### 1. Clone this repository in a folder in your machine.

### 2. Download the Pre-Trained word vectors
After cloning this repository, get the word vectors from [fast text](https://github.com/facebookresearch/fastText/blob/master/docs/pretrained-vectors.md). Download [English Word Vector](https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.en.vec). Create a directory `/wordvectors` in the root directory where the code is cloned. Save the word vector in this directory to be used in next steps.

#### Note: word vector is a huge file. 

### 3. Create required directory for saving intermediate and final outcome of analysis.

Inside wikiconflict directory, create the directory `/data` for storing all the results of analysis.
Inside `/data`, create various subdirectories. Each of these subdirectories will store data at various stages of processing. 
1. `/content`
2. `/change objects`
3. `/change_vector`
4. `/bykau_change_object`
5. `/annotation`
6. `/pre_evaluation`

-----

## Steps of analysis:
As the analysis is done on history of edit of a single article, all the steps of the analysis are performed on a target Wikipedia article. Follow the steps listed below to re-run the analysis:

### 1. Download article
We use tokens from [WikiWho API](https://api.wikiwho.net/en/api/v1.0.0-beta/#/) to identify edited tokens. Therefore, the first step requires to download all the contents of the target article. 
Tokenised content of the article can be downloaded using the [notebook](./notebooks/1_download_rev_content.ipynb) which is saved in the `/data/content` directory for next steps of analysis.

### 2. Create Change Object

From the edited tokens downloaded in the `data/content` directory, we create change vector using the notebook [2_create_change_object-v2.ipynb](./notebooks/2_create_change_object-v2.ipynb). This notebook saves the identified change object in the directory `/data/change` objects.

### 3. Create Change Vector

Next step is to transform Change Objects stored in `/data/change` objects into 600 dimensional Change Vector using pre - trained word vectors downloaded from [fast text](https://github.com/facebookresearch/fastText/blob/master/docs/pretrained-vectors.md). 
The notebook [3_create_change_vector-v3.ipynb](./notebooks/[3_create_change_vector-v3.ipynb]) creates different change vectors corresponding to different values of parameter, *context_length*. All of these change vectors are saved in /`data/change_vector` directory. Change vector is created from neighbouring tokens of change vectors. Neighbouring tokens are tokens in left and right of the gap of change object. Number of neighbouring tokens taken from left and right of change object gap is called *context_length*. Using pre - trained word vectors already downlaoded in `wordvectors`, we average vectors corresponding to  words in neighbouring context. As different values of *context_length* gives differnt neighbouring tokens, we get different Change Vectors for same Change Object for different value of *context_length*.

### 4. Cluster and Evaluate
Change Vectors saved in `/data/change_vector` corresponnding to different values of *context_length* is used to create DBSCAN clusters. These clusters are first evaluated using both intrinsic and extrinsic mechanism. We compare our cluster to one created by re-implementation of [Bykau. et. Al.](https://velgias.github.io/docs/BykauKSV15.pdf). DBSCAN has two parameters *eps* and *min_samples* which when combined with *context_length* gives us three paramters of our model against which we evaluate and analyse performance of our model.

 We use 16 articles in [small article list](https://github.com/acifer/wikiconflict/blob/master/conflicted_article.csv) and [big article list](https://github.com/acifer/wikiconflict/blob/master/conflicted_article-big.csv) for intrinsic evaluation and for comparison with [Bykau. et. Al.](https://velgias.github.io/docs/BykauKSV15.pdf).
 
Finally extrinsic evauation is done on the gold standard created by annotation of change object pertaining to edit history of Wikipedia article [John Logie Baird](https://en.wikipedia.org/wiki/John_Logie_Baird).


#### Intrinsic Evaluation
 
Being a density based clustering algorithm DBSCAN identifies clusters of unequal size. We first analyse the cluster length distribution using various descriptive statistics. We identify Gini co-effecient of cluster length distribution.

For further investigating Change Object groups intrinsicially, we propose various measure based on the assumption that cluster of vector created by averaging word tokens from immediately before and after the Change Object should be able to have similar word tokens and come from similar relative position in the article.  In order to quantify different kind of words in Cluster, we define token entropy for the edited token in gap of change object. Similarly to quantify the relative position of Change Objects in a cluster we define the relative position entropy. All of the intrinsic evaluation analysis is done in the notebook [4_1_clustering-dbscan-intrinsic-evaluation-all.ipynb](./notebooks/4_1_clustering-dbscan-intrinsic-evaluation-all.ipynb). 

To run intrinsic evaluation for all 16 articles we use the script in [3_dbscan_intrinsic.py](./scripts/3_dbscan_intrinsic.py). Results of intrinsic evaluation is saved in `/pre_evaluation` directory. We create visualisation of these results using [notebook](./notebooks/6_2_Plots (Response Variables).ipynb).


#### Compare with [Bykau. et. Al.](https://velgias.github.io/docs/BykauKSV15.pdf)

First we reimplement paper from [Bykau. et. Al.](https://velgias.github.io/docs/BykauKSV15.pdf) Analysis of optimisation and clustering as suggested in [Bykau. et. Al.](https://velgias.github.io/docs/BykauKSV15.pdf) is done in  [5_1_reproduce_fine_grained](./notebooks/5_1_reproduce_fine_grained.ipynb). We run this reimplemented algorithm on all the change objects saved in `/change_object` directory and save the change object groups created by [Bykau. et. Al.](https://velgias.github.io/docs/BykauKSV15.pdf) in `/bykau_change_object`.

####  Agreement of our cluster with [Bykau. et. Al.](https://velgias.github.io/docs/BykauKSV15.pdf)

We find [Fowlkes–Mallows index](https://en.wikipedia.org/wiki/Fowlkes–Mallows_index) of cluster created by our model and [Bykau. et. Al.](https://velgias.github.io/docs/BykauKSV15.pdf) using [5_3_fowlkes_intercluster](./notebooks/5_3_fowlkes_intercluster.ipynb).


#### Extrinsic Evaluation
Extrinsic evaluation is done using [V-Measure Analysis](http://www1.cs.columbia.edu/~amaxwell/pubs/v_measure-emnlp07.pdf). Which is an entropy based measure consisting of Homogenity,Completness and V-measure.

We evaluate our model extriniscally against a golden data set prepared by human annotators. Extrinsic evaluation is done in the [4_2_clustering-dbscan-extrinsic-evaluation-v-measure.ipynb](./notebooks/4_2_clustering-dbscan-extrinsic-evaluation-v-measure.ipynb) for the annotations of the article [John Logie Baird](https://en.wikipedia.org/wiki/John_Logie_Baird). Evaluation is done with respect to three parameters of the model *Context_length*, *eps* and *min_sample*. 

#### Extrinsic evaluation against [Bykau. et. Al.](https://velgias.github.io/docs/BykauKSV15.pdf)
 
 We further evaluate [Bykau. et. Al.](https://velgias.github.io/docs/BykauKSV15.pdf). clusters using annotations of  [John Logie Baird](https://en.wikipedia.org/wiki/John_Logie_Baird). To give a fair comparison, we implement [Bykau. et. al.](https://velgias.github.io/docs/BykauKSV15.pdf) with and without optimisation. [5_2_1_compare-with-bykau_with_optimizations-v-measure.ipynb](./notebooks//5_2_1_compare-with-bykau_with_optimizations-v-measure.ipynb) evaluates Bykau. et. al. with optimisation whereas [5_2_2_compare-with-bykau-without_optimizations-vmeasure](./notebooks/5_2_2_compare-with-bykau-without_optimizations-vmeasure.ipynb).
