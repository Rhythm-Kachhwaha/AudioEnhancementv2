# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['core\\realtime_mic_bridge.py'],
    pathex=[],
    binaries=[],
    datas=[('checkpoint', 'checkpoint'), ('testing/refaudio', 'testing/refaudio'), ('C:\\Users\\mayan\\AppData\\Local\\Programs\\Python\\Python310\\Lib\\site-packages\\triton', 'triton'), ('C:\\Users\\mayan\\AppData\\Local\\Programs\\Python\\Python310\\Lib\\site-packages\\nemo', 'nemo'), ('C:\\Users\\mayan\\AppData\\Local\\Programs\\Python\\Python310\\Lib\\site-packages\\lightning_fabric', 'lightning_fabric'), ('C:\\Users\\mayan\\AppData\\Local\\Programs\\Python\\Python310\\Lib\\site-packages\\torch', 'torch')],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='realtime_mic_bridge',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='realtime_mic_bridge',
)
