import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "boto3"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "statsmodels"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy"])
#subprocess.check_call([sys.executable, "-m", "pip", "install", "fsspec"])
#subprocess.check_call([sys.executable, "-m", "pip", "install", "s3fs"])
#subprocess.check_call([sys.executable, "-m", "pip", "install", "awswrangler"])