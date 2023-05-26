
## A Closer Look at In-Context Learning under Distribution Shifts

[Arxiv Paper](link)

## Setup

1. Install the dependencies using Conda. 

    ```
    conda env create -f environment.yml
    conda activate in-context-learning
    ```

2. Train set-based MLPs. Add you wandb id and path to the directory to save the models in the YAML file (linear_regression_icl2.yaml). 

    ```
    python iclmlp_cln_jmtd.py --config confs/linear_regression_icl2.yaml

    ```

3. Compare set-based MLP to transformers, OLS and tidge. For training a transformer, we use the code here https://github.com/dtsip/in-context-learning. Once both set-based MLP and transformer are trained, specify the paths to their respective checkpoints in the YAML file (comparison_mlp_txformer.yaml) and run.

    ```
    python Comparisons_final.py --config confs/comparison_mlp_txformer.yaml
    ```

## License 

The majority of iclmlp is licensed under CC-BY-NC, however portions of the project (base_models,eval,samplers,tasks) are available under separate license terms: [in-context-learning](https://github.com/dtsip/in-context-learning) is licensed under the MIT license.