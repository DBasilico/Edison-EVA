import pyspark.sql.functions as f
from pyspark.sql.types import *
from pyspark.sql import SparkSession

NOME_TABELLA = 'ANAG_SII'
NOME_BUCKET = 'eva-qa-s3-model'

spark = SparkSession.builder.appName('ANAGRAFICA: da file').enableHiveSupport().getOrCreate()

df = spark.read.option('header', True).option('delimiter', ';').option('inferSchema', False) \
    .csv(f's3://{NOME_BUCKET}/datamodel/ANAG_THOR.csv')

df = df.withColumn('DT_INI_VALIDITA', f.to_date(f.col('DT_INI_VALIDITA'), 'dd/MM/yyyy HH:mm:ss'))
df = df.withColumn('DT_FIN_VALIDITA', f.to_date(f.col('DT_FIN_VALIDITA'), 'dd/MM/yyyy HH:mm:ss'))

df = df.withColumn('DT_INI_FORNITURA', f.to_date(f.col('DT_INI_FORNITURA'), 'dd/MM/yyyy HH:mm:ss'))
df = df.withColumn('DT_FIN_FORNITURA', f.to_date(f.col('DT_FIN_FORNITURA'), 'dd/MM/yyyy HH:mm:ss'))

df = df.withColumn('CONSUMO_ANNUO_COMPLESSIVO', f.regexp_replace(f.col('CONSUMO_ANNUO_COMPLESSIVO'), ',', '.').cast(DoubleType()))
df = df.withColumn('CONSUMO_ANNUO_F1', f.regexp_replace(f.col('CONSUMO_ANNUO_F1'), ',', '.').cast(DoubleType()))
df = df.withColumn('CONSUMO_ANNUO_F2', f.regexp_replace(f.col('CONSUMO_ANNUO_F2'), ',', '.').cast(DoubleType()))
df = df.withColumn('CONSUMO_ANNUO_F3', f.regexp_replace(f.col('CONSUMO_ANNUO_F3'), ',', '.').cast(DoubleType()))

df.write.mode('overwrite').parquet(f's3://{NOME_BUCKET}/datamodel/{NOME_TABELLA}')

