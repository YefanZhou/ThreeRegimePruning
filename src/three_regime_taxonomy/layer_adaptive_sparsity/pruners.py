import numpy as np
from torch.nn.utils import prune
from three_regime_taxonomy.layer_adaptive_sparsity.utils import get_modules

def weight_pruner_loader(config):
    """
    Gives you the pruning methods: 
    """
    if config['prune_layer_sparsity'] == 'glob':
        print("--------------------> use unstructured global pruning")
        return prune_weights_global_unstructured
    elif config['prune_layer_sparsity'] == 'unif':
        print("--------------------> use unstructured uniform pruning")
        return prune_weights_uniform_unstructured


def prune_weights_reparam(model):
    """prune_weights_reparam: Allocate identity mask to every weight tensors.

    Args:
        model (_type_): _description_
    """
    module_list = get_modules(model)
    for m in module_list:
        prune.identity(m,name="weight")

def prune_weights_remove_reparam(model):
    module_list = get_modules(model)
    for m in module_list:
        prune.remove(m,name="weight")


def prune_weights_global_unstructured(model, amount, pruned_layers):
    parameters_to_prune = _extract_weight_tuples(model, pruned_layers)
    prune.global_unstructured(parameters_to_prune, pruning_method = prune.L1Unstructured, amount=amount)

    remained_params_lst = []
    origin_params_lst = []
    module_list = get_modules(model)
    for i, m in enumerate(module_list):
        mask = list(m.named_buffers())[0][1]
        remained_params_lst.append(mask.sum().item())
        origin_params_lst.append(np.prod(mask.shape))

    return np.array(remained_params_lst), np.array(origin_params_lst)

def prune_weights_uniform_unstructured(model, amount, pruned_layers):
    remained_params_lst = []
    origin_params_lst = []
    module_list = get_modules(model)
    assert amount <= 1 # Can be updated later to handle > 1.
    for i, m in enumerate(module_list):
        if i in pruned_layers:
            #print(i, m)
            prune.l1_unstructured(m,name="weight",amount=amount)
        mask = list(m.named_buffers())[0][1]
        remained_params_lst.append(mask.sum().item())
        origin_params_lst.append(np.prod(mask.shape))

    return np.array(remained_params_lst), np.array(origin_params_lst)



"""
These are not intended to be exported.
"""

def _extract_weight_tuples(model, pruned_layers):
    """
    Gives you well-packed weight tensors for global pruning.
    """
    mlist = get_modules(model)
    mlist = [mlist[i] for i in pruned_layers]
    return tuple([(m,'weight') for m in mlist])



