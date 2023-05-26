# A Closer Look At In-Context Learning Under Distribution Shifts

## Setup

1. Install the dependencies using Conda. 

    ```
    conda env create -f environment.yml
    conda activate in-context-learning
    ```

2. Train set-based MLPs. Change the YAML file (linear_regression_icl2.yaml) with your Wandb id and directory to store the models.
    ```
    python iclmlp_cln_jmtd.py --config confs/linear_regression_icl2.yaml

    ```

3. Compare set-based MLP to transformers, OLS and Ridge. For training transformer, we use https://github.com/dtsip/in-context-learning. Specify the path to checkpoints of set-based MLP and transformer model in the YAML file (comparison_mlp_txformer.yaml) and run

    ```
    python Comparisons_final.py --config confs/comparison_mlp_txformer.yaml
    ```
