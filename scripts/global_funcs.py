def load_raw_data_config(config_file_name:str='../data_configs/raw_to_parquet/config.yaml'):
    import yaml

    with open(config_file_name, "r") as ymlfile:
        config = yaml.safe_load(ymlfile)
    return config
    
#     class ConfigError(Exception):
#         """ Custom Error indicating there is an issue with the configuration file """
#         pass
   
#     try:
#         with open(config_file_name, "r") as ymlfile:
#             config = yaml.safe_load(ymlfile)
#         return config
    
#     except:
#         raise ConfigError('There is a syntax error with the configuration file as it is improperly formatted. ' \
#                           'Please refer to the template configuration files and check colons, indentations, quotes, etc.')


def load_data_config(config_file_name:str='../data_configs/config.yaml'):
    import yaml
    
    with open(config_file_name, "r") as ymlfile:
        config = yaml.safe_load(ymlfile)
    config['unq_labs_dir'] = f"{config['output_dir']}/{config['project_name']}/data/unq_labels" 
    config['unq_labs_dir_csv'] = f"{config['output_dir']}/{config['project_name']}/data/unq_labels.csv" 
    config['data_dir'] = f"{config['output_dir']}/{config['project_name']}/data/{config['project_name']}"
    config['nvtab_dir'] = f"{config['output_dir']}/{config['project_name']}/nvtab"
    config['dask_dir'] = f"{config['output_dir']}/{config['project_name']}/dask"
    config['tensorboard_dir'] = f"{config['output_dir']}/{config['project_name']}/tensorboard"
    config['model_checkpoints_dir'] = f"{config['output_dir']}/{config['project_name']}/checkpoints/model_checkpoints"
    config['model_checkpoints_parent_dir'] = f"{config['output_dir']}/{config['project_name']}/checkpoints"
    config['model_weights_dir'] = f"{config['output_dir']}/{config['project_name']}/model_weights.h5"
    return config
    
#     class ConfigError(Exception):
#         """ Custom Error indicating there is an issue with the configuration file """
#         pass
   
#     try:
#         with open(config_file_name, "r") as ymlfile:
#             config = yaml.safe_load(ymlfile)
#         config['unq_labs_dir'] = f"{config['output_dir']}/{config['project_name']}/data/unq_labels" 
#         config['unq_labs_dir_csv'] = f"{config['output_dir']}/{config['project_name']}/data/unq_labels.csv" 
#         config['data_dir'] = f"{config['output_dir']}/{config['project_name']}/data/{config['project_name']}"
#         return config
    
#     except:
#         raise ConfigError('There is a syntax error with the configuration file as it is improperly formatted. ' \
#                           'Please refer to the template configuration files and check colons, indentations, quotes, etc.')
        

def load_model_config(config_file_name:str='../model_configs/model_config.yaml'):
    import yaml
    
    with open(config_file_name, "r") as ymlfile:
        config = yaml.safe_load(ymlfile)
    return config
    
#     class ConfigError(Exception):
#         """ Custom Error indicating there is an issue with the configuration file """
#         pass
   
#     try:
#         with open(config_file_name, "r") as ymlfile:
#             config = yaml.safe_load(ymlfile)
#         return config
    
#     except:
#         raise ConfigError('There is a syntax error with the configuration file as it is improperly formatted. ' \
#                           'Please refer to the template configuration files and check colons, indentations, quotes, etc.')


def get_num_of_classes():
    import pandas as pd
    config = load_data_config()
    filepath = config['unq_labs_dir_csv']
    n_classes = pd.read_csv(filepath).shape[0]
    return n_classes