import sys
import subprocess

subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'tensorflow==2.0.0'])


subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'tensorflow_addons==0.6.0'])