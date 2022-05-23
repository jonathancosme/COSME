def load_program_config(config_file_name:str='config.yaml'):
    import yaml
    
    class ConfigError(Exception):
        """ Custom Error indicating there is an issue with the configuration file """
        pass
   
    try:
        with open(config_file_name, "r") as ymlfile:
            config = yaml.safe_load(ymlfile)
        return config
    
    except:
        raise ConfigError('There is a syntax error with the configuration file as it is improperly formatted. '
                          'Please refer to the template configuration files and check colons, indentations, quotes, etc.')