from PyInstaller.utils.hooks import collect_data_files
import os
import smolagents

# Get the base path to the smolagents package
base = os.path.dirname(smolagents.__file__)

# Collect only .yaml files from the 'prompts' subfolder
yaml_files = []
prompts_path = os.path.join(base, 'prompts')

for fname in os.listdir(prompts_path):
    if fname.endswith('.yaml'):
        full_path = os.path.join(prompts_path, fname)
        yaml_files.append((full_path, 'smolagents/prompts'))

# Tell PyInstaller to include those files
datas = yaml_files
