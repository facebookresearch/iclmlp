from quinine import (tstring,
    tinteger,
    tfloat,
    tboolean,
    stdict,
    tdict,
    default,
    required,
    allowed,
    nullable,
)
from funcy import merge


model_schema = {
    "input_dim": merge(tinteger, required), 
    "psi_hidden_dim": merge(tinteger, required), 
    "rho_hidden_dim": merge(tinteger, required), 
    "psi_output_dim": merge(tinteger, required), 
    "rho_output_dim": merge(tinteger, required), 
    "xi_hidden_dim": merge(tinteger, required), 
    "output_dim": merge(tinteger, required), 
    "lr":merge(tfloat, required), 
    "num_epochs":merge(tinteger, required),
    "activations": merge({"type": "list", "schema": tstring}, required),
    "batch_norms": merge({"type": "list", "schema": tboolean}, required),
    "layer_norms": merge({"type": "list", "schema": tboolean}, required),
    "dropouts": merge({"type": "list", "schema": tfloat}, required),
    "architecture": merge(tstring, required)
}



data_schema = {
    "num_examples": merge(tinteger, required), 
    "data_dim": merge(tinteger, required), 
    "prompt_max_len_train": merge(tinteger, required),
    "prompt_max_len_test": merge(tinteger, required),
    "noise": merge(tboolean, required)
}



schema = {
    "model": stdict(model_schema),
    "data": stdict(data_schema),
    "out_dir": merge(tstring, required),
    "wandb_entity":merge(tstring, required)
}