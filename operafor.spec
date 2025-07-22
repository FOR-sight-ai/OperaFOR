# -*- mode: python ; coding: utf-8 -*-
import os
from PyInstaller.utils.hooks import collect_data_files, copy_metadata

block_cipher = None

# Collect metadata for packages that need it
datas = [('index.html', '.')]
datas += copy_metadata('fastmcp')
datas += copy_metadata('mcp')
datas += copy_metadata('smolagents')

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=['aiohttp', 'smolagents.prompts', 'fastmcp', 'mcp'],
    hookspath=['hooks'],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries + a.zipfiles + a.datas,  
    exclude_binaries=False,             
    name='operafor',
    debug=False,                        
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,                       
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    onefile=True,                       
)
