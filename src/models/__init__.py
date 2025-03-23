from .net import Net

_model_dict = {
    "net": Net
}

def get_model(model_config):
    model_name = model_config.model_name
    Model = _model_dict[model_name]
    return Model(**model_config.model_config)