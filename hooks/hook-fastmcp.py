from PyInstaller.utils.hooks import copy_metadata, collect_data_files

# Copy package metadata for fastmcp
datas = copy_metadata('fastmcp')

# Collect any data files that fastmcp might need
datas += collect_data_files('fastmcp')

# Ensure all fastmcp submodules are included
hiddenimports = [
    'fastmcp.core',
    'fastmcp.server', 
    'fastmcp.tools',
    'fastmcp.resources',
    'fastmcp.prompts',
]
