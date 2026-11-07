"""Capture software versions without user names, host names or credentials."""
import platform
import subprocess
from importlib import metadata


def environment_info():
    packages = {}
    for name in ('h2rbench','Pillow','requests','openai','google-genai','torch','pyiqa','numpy','opencv-python'):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    try:
        ffmpeg = subprocess.check_output(['ffmpeg','-version'],text=True).splitlines()[0]
    except (OSError, subprocess.CalledProcessError):
        ffmpeg = None
    return dict(python=platform.python_version(), system=platform.system(),
                release=platform.release(), architecture=platform.machine(),
                packages=packages, ffmpeg=ffmpeg)
