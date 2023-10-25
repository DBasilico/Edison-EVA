from pyspark.sql.types import *
from pyspark.sql import SparkSession
import pandas as pd
from datetime import datetime

NOME_TABELLA = 'CONFIGURAZIONE_CLUSTER'
NOME_BUCKET = 'eva-qa-s3-model'

spark = SparkSession.builder.appName('CONFIGURAZIONE CLUSTER: da script').enableHiveSupport().getOrCreate()

schema = StructType([
    StructField('SIMULAZIONE', StringType(), False),
    StructField('REGOLA_SELECT', StringType(), True),
    StructField('LISTA_POD_SINGOLI', ArrayType(StringType(), False), True),
    StructField('CLUSTER_POD_SINGOLI', ArrayType(StringType(), False), True),
    StructField('CLUSTER_POD_SINGOLI_NOME', ArrayType(StringType(), False), True),
    StructField('REGOLA_WHERE', StringType(), True),
    StructField('ORDINAMENTO', IntegerType(), False),
    StructField('VALIDITA', TimestampType(), False)
])

### Recuperiamo i pod singoli da file excel ###

lista_pod = list(pd.read_excel('C:/Users/dbasilico/Desktop/Progetti/Edison/EVA/Edison-EVA-sviluppi/list_pod.xlsx')['Pod'].unique())

now = datetime.now()

data = [
    ['1000', "concat_ws('#', coalesce(ZONA,''), coalesce(ID_AREA_GESTIONALE,''), '')", None, None, None, None, 1, now],
    ['1000', None, lista_pod, None, None, None, 2, now],
    ['1001', "concat_ws('#', coalesce(ZONA,''), coalesce(ID_AREA_GESTIONALE,''), '')", None, None, None, None, 1, now],
    ['1001', None, lista_pod, None, None, None, 2, now]
]

spark.createDataFrame(data, schema).write.mode('overwrite').parquet(f'./{NOME_TABELLA}')

