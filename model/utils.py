import yaml


def load_yaml_config(file_path):
    """
    Load yaml configuration file

    :param file_path: Path to the yaml configuration file
    :return: Contents of the configuration file
    """
    with open(file_path, "r") as file:
        config = yaml.safe_load(file)
    print(config)
    return config


if __name__ == "__main__":
    load_yaml_config("./config.yml")