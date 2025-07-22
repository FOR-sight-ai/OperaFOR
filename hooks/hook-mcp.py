from PyInstaller.utils.hooks import copy_metadata, collect_data_files

# Copy package metadata for mcp
datas = copy_metadata('mcp')

# Collect any data files that mcp might need
datas += collect_data_files('mcp')

# Ensure all mcp submodules are included
hiddenimports = [
    'mcp.client',
    'mcp.server',
    'mcp.types',
    'mcp.shared',
]
