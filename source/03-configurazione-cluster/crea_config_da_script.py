import pyspark.sql.functions as f
from pyspark.sql.types import *
from pyspark.sql import SparkSession
import pandas as pd
import boto3
from datetime import datetime

NOME_TABELLA = 'CONFIGURAZIONE_CLUSTER'
NOME_BUCKET = 'eva-qa-s3-model'

spark = SparkSession.builder.appName('CONFIGURAZIONE CLUSTER: da script').enableHiveSupport().getOrCreate()

schema = StructType([
    StructField('SIMULAZIONE', StringType(), False),
    StructField('REGOLA_SELECT', StringType(), True),
    StructField('LISTA_POD', ArrayType(StringType(), False), True),
    StructField('REGOLA_WHERE', StringType(), True),
    StructField('ORDINAMENTO', IntegerType(), False),
    StructField('VALIDITA', TimestampType(), False)
])

### Recuperiamo i pod singoli da file excel ###

lista_pod = list(pd.read_excel('C:/Users/dbasilico/Desktop/Progetti/Edison/EVA/Edison-EVA-sviluppi/Dati/lista_pod.xlsx')['Pod'].unique())

regola_default = "concat_ws('#', coalesce(ZONA,''), coalesce(ID_AREA_GESTIONALE,''), case when ID_AREA_GESTIONALE='RES' then coalesce(PROVINCIA_CRM,'') else '' end)"

now = datetime.now()

data = [
    ['1000', regola_default, None, None, 1, now],
    ['1000', None, lista_pod, None, 2, now]
]

spark.createDataFrame(data, schema).write.mode('overwrite').parquet('C:/Users/dbasilico/Desktop/Progetti/Edison/EVA/Edison-EVA-sviluppi/CONFIGURAZIONE_CLUSTER')

