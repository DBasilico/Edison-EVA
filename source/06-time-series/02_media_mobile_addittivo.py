import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from statsmodels.tsa.seasonal import seasonal_decompose

df = pd.read_parquet('C:/Users/dbasilico/Desktop/Progetti/Edison/EVA/Edison-EVA-sviluppi/BEST_CHOICE.parquet')

CLUSTERS = df.CL


df = df[df.CLUSTER == ID_CLUSTER]
df = df[df.TIMESTAMP > datetime(2023,1,1)]
df = df.set_index('TIMESTAMP').sort_index()

df = df[['ENERGIA_2G_BEST']]
