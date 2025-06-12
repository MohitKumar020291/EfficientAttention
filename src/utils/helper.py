def read_yaml(file_path):
    import yaml
    """Reads a YAML file and returns its content as a Python dictionary."""
    try:
        with open(file_path, 'r') as file:
            yaml_content = yaml.safe_load(file)
        return yaml_content
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        return None