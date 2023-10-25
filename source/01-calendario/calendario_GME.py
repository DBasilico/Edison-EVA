import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "holidays"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas"])

from datetime import datetime, date
import holidays
import pytz
from pyspark.sql import functions as f
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, Column, create_map, lit, when, isnull
from pyspark.sql.dataframe import DataFrame as sparkDF
from itertools import chain
from pyspark.sql import Window

NOME_TABELLA = 'CALENDARIO_GME'
NOME_BUCKET = 'eva-qa-s3-model'

def apply_map(
        df: sparkDF, col_name: str, map_dict: dict, na_replace=None,
        mantieni_mancanti: bool = True, default_mancanti=None,
        output_col_name: str = None, output_format=None
):
    if not isinstance(col_name, Column):
        col_name = col(col_name)
    mapping_expr = create_map([lit(x) for x in chain(*map_dict.items())])
    if pd.isna(output_col_name):
        output_col_name = col_name._jc.toString()
    if mantieni_mancanti:
        expr = when(~isnull(mapping_expr[col_name]), mapping_expr[col_name]).otherwise(col_name)
    else:
        expr = when(~isnull(mapping_expr[col_name]), mapping_expr[col_name]).otherwise(lit(default_mancanti))
    if not pd.isna(output_format):
        expr = expr.cast(output_format)
    df = df.withColumn(output_col_name, expr)
    if na_replace is not None:
        df = df.fillna(na_replace, subset=[output_col_name])
    return df

spark = SparkSession.builder.appName('CALENDARIO v1.0.0').enableHiveSupport().getOrCreate()

eng2ita_mapdays = {
    'Mon': 'Lun',
    'Tue': 'Mar',
    'Wed': 'Mer',
    'Thu': 'Gio',
    'Fri': 'Ven',
    'Sat': 'Sab',
    'Sun': 'Dom'
}

min_anno = 2015

df_finale = None

for anno in range(min_anno, datetime.now().year+1):
    print(anno)
    festivita = [date(x.year, x.month, x.day) for x in holidays.IT(years=range(anno, anno+1)).keys()]
    F3_cond = (f.col('DATA').isin(festivita)) | \
              (f.col('GIORNO_SETTIMANA') == 'Dom') | \
              ((f.col('ORA_GME')-f.lit(1)).isin([0, 1, 2, 3, 4, 5, 6, 23]))
    F2_cond = ((f.col('GIORNO_SETTIMANA') == 'Sab') & ((f.col('ORA_GME')-f.lit(1)).isin(list(range(7, 23))))) | \
              ((f.col('ORA_GME')-f.lit(1)).isin([7, 19, 20, 21, 22]))
    PICCO_cond = (f.col('DATA').isin(festivita)) | \
                 (f.col('GIORNO_SETTIMANA').isin(['Sab', 'Dom'])) | \
                 ((f.col('ORA_GME')-f.lit(1)).isin(list(range(0, 8))+list(range(20, 24))))
    df = pd.DataFrame(index=pd.date_range(start=f'1/1/{anno} 00:00:00', end=f'31/12/{anno} 23:00:00',
                                      freq='H', tz='Europe/Rome'))
    df['UNIX_TIME'] = [date.timestamp() for date in df.index]
    df['TIMESTAMP'] = [date.strftime('%Y-%m-%d %H:%M:%S %z') for date in df.index]
    df = spark.createDataFrame(df)
    df = df.withColumn('UNIX_TIME', f.col('UNIX_TIME').cast('bigint'))
    df = df.withColumn('DATA', f.to_date(f.col('TIMESTAMP')))
    df = df.withColumn('GIORNO', f.dayofmonth(f.col('DATA')))
    df = df.withColumn('MESE', f.month(f.col('DATA')))
    df = df.withColumn('ANNO', f.year('DATA'))
    df = df.withColumn('NUMERO_SETTIMANA', f.weekofyear(f.col('DATA')))
    df = df.withColumn('GIORNO_SETTIMANA', f.date_format(f.col('DATA'), 'E'))
    df = apply_map(df, col_name='GIORNO_SETTIMANA', map_dict=eng2ita_mapdays,
                   mantieni_mancanti=False, default_mancanti='ERRORE')
    w_GME = Window.partitionBy('ANNO', 'MESE', 'GIORNO').orderBy(f.col('UNIX_TIME').asc())
    df = df.withColumn('ORA_GME', f.rank().over(w_GME))
    df = df.withColumn('FASCIA', f.when(F3_cond, f.lit('F3'))
                       .when(F2_cond, f.lit('F2'))
                       .otherwise(f.lit('F1'))
                       )
    df = df.withColumn('PKOP', f.when(PICCO_cond, f.lit('OP'))
                       .otherwise(f.lit('PK'))
                       )
    if df_finale is None:
        df_finale = df
    else:
        df_finale = df_finale.unionByName(df)

#df_finale.repartition(32, 'UNIX_TIME').write.partitionBy(['ANNO']).mode('overwrite').parquet(f's3://{NOME_BUCKET}/datamodel/{NOME_TABELLA}')
df_finale.repartition(32, 'UNIX_TIME').write.partitionBy(['ANNO']).mode('overwrite').parquet(f'./CALENDARIO_GME')
