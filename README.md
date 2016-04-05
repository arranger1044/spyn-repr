# spyn-repr
Code for representation learning experiments with Sum-Product Networks

## Requirements

### Python packages
The following python 3 packages have been used:
```
numpy 1.10.4
scipy 0.16.1
sklearn 0.17
pandas 0.15.2
theano 0.8.0
matplotlib 1.5.1
seaborn 0.6.0
```

### Numerical libraries
For optimal `theano` and `numpy` performances one can exploit the
`blas` and `lapack` libs. CUDA is required to use the GPU with Theano.
To properly install them please refer to
[this page](http://deeplearning.net/software/theano/install.html)
The lib versions used are:
```
liblapack3 3.5.0-2
libopenblas-dev 0.2.8-6
cuda 6.0.1
```

### Libra toolkit
The [Libra toolkit version 1.0.1](http://libra.cs.uoregon.edu/) has
been used to learn the Mixture of Trees (MT).

### Commands execution
`ipython 3.2.1` has been used to launch all scripts.
The following commands will assume the use of `ipython` and being in
the repo main directory:
```
cd spyn-repr
```

## Data
The dataset used are provided in the `data` folder.
The **X** values only are stored as compressed text files;
The complete versions, containing also the class information, are
stored as pickle files in `data/<dataset-name>` sub-folders.
To uncompress them run:
```
tar -zxvf ocr_letters.data.tgz
tar -zxvf caltech101.data.tgz
tar -zxvf bmnist.data.tgz
```

## Learning Models
In the `models` dir the SPN and MT models employed in the
experiments are provided as a compressed archives: `<dataset-name>.models.tbz2`.
To uncompress them:
```
cd models
tar -jxvf bmnist.models.tbz2
tar -jxvf caltech101.models.tbz2
tar -jxvf ocr_letters.models.tbz2
```
Three directories will be created, one for each dataset.

The following sub sections will list the commands to learn them back from data.

### Learning SPNs
To learn an SPN structure with `LearnSPN-b` one can use the script
`learnspn_exp.py` in the `bin` directory, after specifying a dataset
name.

The G-Test threshold parameter values can be specified with `-g`.
The stopping criterion values for the min number of instances to split
is specified via the `-m` option.
The smoothing coefficient values can be specified through the `-a`
option.

For instance, to learn the SPN-I model on `ocr_letters` used in the
paper, and saving its output in `exp/learnspn/` run the following command:
```
ipython -- bin/learnspn_exp.py ocr_letters -k 2 -c GMM -g 15 -m 500 -a 0.1 0.2 0.5 1.0 2.0 -v 1 -o exp/learnspn/ --save-model
```

### Learning MTs
The script `mtlearn_exp.py` as a python wrapper to use the `mtlearn`
command from Libra.
It is necessary to specify the path to the libra installation bin dir
through the parameter `-e`.
For instance to learn the MT-I model, comprising 3 mixtures
components, one shall exectute:
```
ipython --  bin/mtlearn_exp.py ocr_letters -n 3 3 -e /root/Desktop/libra-tk-1.0.1/bin/ -o exp/mtlearn/ -i 1
```
The model output will be stored in the dir specified by `-o`. The
model path that later will be used to extract the embeddings shall
point to the AC representation of the mixtures: the `.ac` files



## Visualizations
Use the `visualize_spn.py` script to reproduce the visualizations
pictured in the paper. To save the outputs use the options: `--save
pdf -o <output-path>`.
The following commands show how to set some parameters to reproduce
the figures and plot shown in the paper.

Some common options among the commands are: `--size N M` to specify
how to render the images as matrices (for CAL and BMN `M` = `N` = 28 (default),
for OCR `M` = 16 and `N` = 8). `--n-cols K` determines to display the
images in a grid of `K` columns.
`--max-n-images` limits the output of a command to a number of images.
`--invert` inverts the black and
white in the displayed images; `--space` determines the horizontal and
vertical space among displayed images, when in a grid.
### Visualizing samples

To generate and visualize samples from a learned SPN, run a command
like this one:
```
ipython -- bin/visualize_spn.py bmnist --model
models/bmnist/bmnist_spn_50/best.bmnist.model --sample 9 --size 28 28
--nn --n-cols 3 --invert --space 0.0 0.0
```
where `--model` determines the path to the learned model; `--sample`
specifies the number of samples to generate and `--nn` specifies
whether to show the training nearest neighbor images.

### Visualizing node activations
To plot the activations for an SPN nodes given a query image, run a
command like this one:
```
ipython -- bin/visualize_spn.py bmnist --model
models/bmnist/bmnist_spn_100/best.bmnist.model --activations 0 --invert --space 0.0 0.0
```
where `--activations Y` sets the id of the query instance to display.

### Visualizing marginal inference

To reproduce the marginal inference visualizations given a series of
SPN models and a query image, run a command like this one:
```
ipython -- bin/visualize_spn.py bmnist --model models/bmnist/bmnist_spn_500/best.bmnist.model models/bmnist/bmnist_spn_100/best.bmnist.model models/bmnist/bmnist_spn_50/best.bmnist.model --marg-activations 0  4000 5000 10000 20000  --invert
```
where  `--model [path]+` accepts a list of paths to the desired models and
`--marg-activations Y+` sets the the ids of the query instances to display.
image
### Visualizing mpe filters
To visualize the learned filters for the nodes by exploiting MPE
inference, use the `--mpe` option.
To reproduce a visualization by scope length used in the paper, run
something like this:
```
ipython -- bin/visualize_spn.py bmnist --model models/bmnist/bmnist_spn_50/best.bmnist.model --mpe scope --scope-range 10 100 --invert --n-cols 3 --max-n-images 9
```
where `--mpe scope` determines the visualization by scope length and
`--scope-range` actually specify a range of scope lengths.

### Visualizing collapsed MPE clusters

To visualize the groups of instances with the same MPE traversal tree
path, use the option `--hid-groups`, like in this way:
```
ipython -- bin/visualize_spn.py bmnist --hid-groups /media/valerio/formalità/repr/bmnist/mpe/500-mpe-hid-var.bmnist.pickle -1 --size 28 28 --n-cols 3 --max-n-images 9
```
when `--hid-groups` specifies the path to a pickle containing the
representation embedding splits as computed by `spn_repr_exp.py`

### Visualizing scope length distributions
To plot the scope length distribution of a learned SPN model, use the
`--scope hist` option, optionally limiting the graph on the y and x axis:
```
ipython -- bin/visualize_spn.py caltech101 --model models/caltech101/caltech101_spn_100/best.caltech101.model --scope hist --ylim 200000 --xlim -10 785
```

To visualize the bar graphs for the scope length distributions
layer-wise, as shown in the supplemental materials, run with the
`--scope lmap` option, as in:
```
ipython -- bin/visualize_spn.py bmnist --model models/bmnist/bmnist_spn_50/best.bmnist.model --scope lmap
```
## Extracting Embeddings
Given a dataset split into train, validation and test, the embedding
generation functions will produce the new train, validation and test
splits according to a model and some filtering criterion. In addition
to that, for SPN models, a feature file map will be generated,
comprising information about the node used to generate each feature.

### Extracting SPN embeddings
To extract the embeddings for a dataset from an SPN model one can use
the `spn_repr_data.py` script. He will need to specify some string
matching rules to identiy the dataset splits with `--train-ext`,
`--valid-ext`, `--test-ext` and the directory where to look as the
first parameter (e.g. `data/`).
The SPN model path is specified with the option `--model` and the new
representation name will be composed using the `--suffix` parameter
value.
To specify how to extract the embeddings, one has to set two options:
`--ret-func` which determines which values to extract from a node (to
get the node output value in the log domain use `"var-log-val"`), and
`--filter-func` that indicates which nodes to consider to generate the
embeddings (set it to `"all"` to get all nodes with scope length
greater than 1, as in the experiments).

Here is an example usage:
```
ipython -- bin/spn_repr_data.py data/ --train-ext ocr_letters.ts.data --valid-ext ocr_letters.valid.data --test-ext ocr_letters.test.data --model models/ocr_letters/ocr_letters_spn_100/best.ocr_letters.model -o /media/valerio/formalità/repr/ocr_letters/ --ret-func "var-log-val" --filter-func "all" --suffix "100-all-log-val" --no-ext --no-mpe --fmt float
```

To extract the embeddings for the MPE tree path visualization, run
this other version:
```
ipython -- bin/spn_repr_data.py data/ --train-ext bmnist.ts.data --valid-ext bmnist.valid.data --test-ext bmnist.test.data --model models/bmnist/bmnist_spn_500/best.bmnist.model -o /media/valerio/formalità/repr/bmnist/mpe/ --ret-func "max-var" --filter-func "hid-var" --suffix "500-mpe-hid-var" --no-ext --fmt int
```
which uses only the max child branches for each hidden r.v. (specified
with `--filter-func "hid-var"`) and sets them to 0 or 1 (specified with `--ret-func "max-var"`).


#### Filtering SPN embeddings
One can extract embeddings comprising all (non-leaf) nodes, then
filter them without running again `spn_repr_data.py`. To do so, use
the `filter_feature_repr.py` script.
```
ipython -- bin/filter_feature_repr.py /media/valerio/formalità/repr/caltech101/all/ -r 50-all-log-val.caltech101 --train-ext ts.data --valid-ext valid.data --test-ext test.data --info /media/valerio/formalità/repr/caltech101/all/50-all-log-val.caltech101.features.info -o /media/valerio/formalità/repr/caltech101/non-leaf/ --suffix 50-non-leaf-log-val-caltech101 --save-text --nodes sum prod
```

### Extracting RBM embeddings
To learn an RBM with `sklearn` and use the model to generate the new
embeddings for the initial dataset, use the `rbm_repr_data.py` script:
```
ipython -- bin/rbm_repr_data.py data/ --train-ext bmnist.ts.data
--valid-ext bmnist.valid.data --test-ext bmnist.test.data  --suffix
"rbm-1000" --no-ext --fmt float --n-hidden 1000 --l-rate 0.1 0.01
--batch-size 20 100 --log -v 2 --n-iters 10 20 30
```
in which `--n-hidden` specifies the number of hidden units, `--l-rate`
sets the learning rate values, `--batch-size` the batch size
parameters and `--n-iters` the numbers of epochs. The option `--log`
let the resulting data to be saved in the log domain as well, with the
name `log.rbm-1000`

### Random Marginal Query Feature Generation
The 1000 feature masks employed in the experiments using the marginal
queries can be found as text files under the dir `features`.

To generate them back run the following commands using the
`marg_feature_gen.py` script:
```
ipython -- bin/marg_feature_gen.py caltech101 -o features/caltech101/ --suffix "1000-2-2-10-10-rand-rect" --rand-marg-rect 1000 2 2 10 10
ipython -- bin/marg_feature_gen.py bmnist -o features/bmnist/ --suffix "1000-2-2-10-10-rand-rect" --rand-marg-rect 1000 2 2 10 10
ipython -- bin/marg_feature_gen.py ocr_letters -o features/ocr_letters/ --suffix "1000-2-2-7-7-rand-rect" --rand-marg-rect 1000 2 2 7 7
```

Then, to employ these features to generate the embeddings one has to
invoke the `spn_repr_data.py` and `libra_repr_data` by specifying the
`--features` option in a command as
follows:
```
ipython -- bin/libra_repr_data.py data/ --train-ext ocr_letters.ts.data --valid-ext ocr_letters.valid.data --test-ext ocr_letters.test.data --model exp/mtlearn/ocr_letters_2016-03-01_07-53-35/models/ocr_letters_2_0.ac  -o repr/rect/ocr_letters/ --suffix "3-mt-1000-2-2-7-7-rect" --no-ext --fmt float --features features/ocr_letters/1000-2-2-7-7-rand-rect.ocr_letters.features --acquery-path "/root/Desktop/libra_exp/bin/acquery"
```

#### Splitting and merging features masks
The evaluation of the random queries can be highly time consuming. For
this reason it could be convenient to split the initial feature set
into smaller parts to be computed in parallel.
To split a feature set into smaller sets of size 100, use the
`feature_split` command:
```
ipython -- bin/feature_split.py features/caltech101/1000-2-2-10-10-rand-rect.caltech101.features --split 100
```
After that 10 different feature set files will be generated and can be
used by running `spn_repr_data.py` 10 times.

Then, if one wants to merge the representation splits of those 10
runs, he can use the command `merge_repr.py` in this way:
```
ipython -- bin/merge_repr.py <paths-to-repr-splits> -o
repr/rect/caltech101/ --suffix all-spn-1000-2-2-10-10-rect.caltech
```
Obtaining a single pickle containing the merged splits for training,
validation and test.

#### Running on the GPU
```
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 ipython -- bin/spn_repr_data.py data/ --train-ext caltech101.ts.data --valid-ext caltech101.valid.data --test-ext caltech101.test.data --model models/caltech101/caltech101_spn_500/best.caltech101.model -o /media/valerio/formalità/repr/rect/caltech101/ --suffix "500-spn-1000-2-2-10-10-rect" --no-ext --no-mpe --fmt float --features features/caltech101/1000-2-2-10-10-rand-rect.caltech101.features.0.99 --theano 100 --opt-unique --max-nodes-layer 50
```

## Evaluate Embeddings

To employ the learned representation in the classification tasks, use
the `classify_repr_exp.py` script. It takes as arguments the name of
the dataset and other parameters to determine the classifier
configuration and specify its grid search.

To reproduce the experiments use the same ovr classifier by specifying
the option `--logistic "l2-ovr-bal"` and the regularization values
with `--log-c 0.0001 0.001 0.01 0.1 1.0`.

For instance, to train and evaluate a classifier on `bmnist` on the
representations split and stored in the file ` repr/bmnist/non-leaf/
50-non-leaf-log-val.bmnist.pickle`, run the command:
```
ipython -- bin/classify_repr_exp.py bmnist -r 50-non-leaf-log-val.bmnist --repr-dir repr/bmnist/non-leaf/ --splits .ts.data .valid.data .test.data --dtype float --logistic "l2-ovr-bal" --log-c 0.0001 0.001 0.01 0.1 1.0
```

To reproduce the last experiment, use the additional `--feature-inc 100` option
to learn a classifier on several feature sets incremented by 100 at a
time. E.g.:
```
ipython -- bin/classify_repr_exp.py bmnist -r 50-non-leaf-log-val.bmnist --repr-dir repr/bmnist/non-leaf/ --splits .ts.data .valid.data .test.data --dtype float --logistic "l2-ovr-bal" --log-c 0.0001 0.001 0.01 0.1 1.0
 --feature-inc 100
```
