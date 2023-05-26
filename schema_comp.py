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





schema = {
    "path_mlp": merge(tstring, required),
    "path_txformer": merge(tstring, required),
    "wandb_entity": merge(tstring, required)
}