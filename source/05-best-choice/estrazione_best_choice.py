from pyspark.sql import functions as f
from datetime import date
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName('ESTRAZIONE BEST CHOICE').enableHiveSupport().getOrCreate()

SIMULAZIONE = '1000'
DATA_INIZIO = date(2022, 1, 1)
DATA_FINE = date(2023, 9, 17)

NOME_TABELLA = 'BEST_CHOICE'
NOME_BUCKET = 'eva-qa-s3-model'

df = spark.read.parquet(f's3://{NOME_BUCKET}/datamodel/{NOME_TABELLA}')

df = df.filter(f.col('SIMULAZIONE') == SIMULAZIONE)
df = df.filter(f.col('DATA').between(DATA_INIZIO, DATA_FINE))

df.repartition(1).write.mode('overwrite').parquet(f's3://{NOME_BUCKET}/datamodel/SCARICO_DATI/')


