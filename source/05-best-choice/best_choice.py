import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "boto3"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas"])

from pyspark.sql.types import *
from datetime import datetime
from pyspark.sql import Window
from pyspark.sql import functions as f
from pyspark.sql import SparkSession
from typing import Iterable
from pyspark.sql.dataframe import DataFrame as sparkDF
import boto3
from dateutil.relativedelta import relativedelta
import sys
import pandas as pd


def check_path_s3(path: str, is_file: bool = False):
    if len(path) > 0:
        if path[-1] != '/' and not is_file: path = f'{path}/'
        if path[0] == '/': path = path[1:]
    return path


def path_exists_s3(bucket: str, path: str, is_file: bool):
    path = check_path_s3(path, is_file)
    files = list(boto3.session.Session().resource('s3').Bucket(bucket).objects.filter(Prefix=path))
    return len(files) > 0


def delete_folder_s3(bucket: str, folder_path: str):
    if path_exists_s3(bucket=bucket, path=folder_path, is_file=False):
        folder_path = check_path_s3(folder_path)
        boto3.resource('s3').Bucket(bucket).objects.filter(Prefix=folder_path).delete()


def melt(
        df: sparkDF,
        id_vars: Iterable[str], value_vars: Iterable[str],
        var_name: str = "variable", value_name: str = "value"
) -> sparkDF:
    _vars_and_vals = f.array(*(f.struct(f.lit(c).alias(var_name), f.col(c).alias(value_name)) for c in value_vars))
    _tmp = df.withColumn("_vars_and_vals", f.explode(_vars_and_vals))
    cols = id_vars+[f.col("_vars_and_vals")[x].alias(x) for x in [var_name, value_name]]
    return _tmp.select(*cols)


spark = SparkSession.builder.appName('BEST_CHOICE v1.0.0').enableHiveSupport().getOrCreate()

COD_SIMULAZIONE = sys.argv[1] #'1000'
NOME_TABELLA = 'BEST_CHOICE'
NOME_BUCKET = 'eva-qa-s3-model'
TRACE_ANALISI = eval(sys.argv[2]) #False

INIZIO_PERIODO = datetime.strptime(sys.argv[3], '%Y-%m-%d').date()
FINE_PERIODO = datetime.strptime(sys.argv[4], '%Y-%m-%d').date()
tab_path = f'datamodel/{NOME_TABELLA}/'

data_corrente = INIZIO_PERIODO

##############################
### CONFIGURAZIONE CLUSTER ###
##############################

df_conf_cluster = spark.read.parquet(f's3://{NOME_BUCKET}/datamodel/CONFIGURAZIONE_CLUSTER')
#df_conf_cluster = spark.read.parquet('./CONFIGURAZIONE_CLUSTER')

df_conf_cluster = df_conf_cluster.filter(f.col('SIMULAZIONE') == COD_SIMULAZIONE)

max_validita = df_conf_cluster.agg(f.max('VALIDITA').alias('MAX_VALIDITA')).collect()[0]['MAX_VALIDITA']
df_conf_cluster = df_conf_cluster.filter(f.col('VALIDITA') == max_validita)

df_conf_cluster = df_conf_cluster.pandas_api().set_index('ORDINAMENTO')
df_conf_cluster.sort_index(inplace=True)

if not df_conf_cluster.index.is_unique:
    raise AttributeError('Presente ORDINAMENTO con valori multipli')

while data_corrente <= FINE_PERIODO:

    print(data_corrente)

    ##################
    ### ANAGRAFICA ###
    ##################

    # POD attivi in cui dobbiao calcolare la BEST CHOICE (DATA FORNITURA)
    df_anagrafica = spark.read.parquet(f's3://{NOME_BUCKET}/datamodel/ANAG_SII')
    #df_anagrafica = spark.read.parquet('./ANAG_SII')

    df_anagrafica = df_anagrafica.filter(f.col('TRATTAMENTO') == 'O')

    df_anagrafica = df_anagrafica.withColumn('DT_INI_VALIDITA', f.to_timestamp(f.col('DT_INI_VALIDITA'), 'dd/MM/yyyy HH:mm:SS'))
    df_anagrafica = df_anagrafica.withColumn('DT_FIN_VALIDITA', f.to_timestamp(f.col('DT_FIN_VALIDITA'), 'dd/MM/yyyy HH:mm:SS'))
    df_anagrafica = df_anagrafica.withColumn('CONSUMO_ANNUO_COMPLESSIVO', f.regexp_replace(f.col('CONSUMO_ANNUO_COMPLESSIVO'), ',', '.').cast(DoubleType()))

    # Filtriamo solo i pod che hanno inizio e fine fornitura nella data corrente
    df_anagrafica = df_anagrafica.filter(
        (f.col('DT_FIN_VALIDITA') >= data_corrente) &
        (f.col('DT_INI_VALIDITA') <= data_corrente)
    )

    #if df_anagrafica.groupBy('POD').count().filter(f.col('count') > 1).count() > 0:
    #    raise AttributeError(f'ERRORE: in anagrafica sono presenti pod multipli per il giorno {data_corrente}')

    ###############
    ### CLUSTER ###
    ###############

    df_anagrafica = df_anagrafica.withColumn('CLUSTER', f.lit(None).cast(StringType()))

    for k_rule in df_conf_cluster.index.to_numpy():

        row = df_conf_cluster.loc[int(k_rule)]
        lista_pod_singoli = row['LISTA_POD_SINGOLI']
        cluster_pod_singoli = row['CLUSTER_POD_SINGOLI']
        cluster_pod_singoli_nome = row['CLUSTER_POD_SINGOLI_NOME']
        regola_select = row['REGOLA_SELECT']
        regola_where = row['REGOLA_WHERE']

        if (lista_pod_singoli is not None) and ((cluster_pod_singoli is not None) or (cluster_pod_singoli_nome is not None) or (regola_select is not None) or (regola_where is not None)):
            raise AttributeError('ERRORE: presente LISTA POD SINGOLI assieme a informazioni incompatibili')
        if (cluster_pod_singoli is None) and (cluster_pod_singoli_nome is not None):
            raise AttributeError('ERRORE: presente NOME CLUSTER POD SINGOLI ma manca LISTA')
        if (cluster_pod_singoli is not None) and (cluster_pod_singoli_nome is None):
            raise AttributeError('ERRORE: presente CLUSTER POD SINGOLI ma manca NOME')
        if (cluster_pod_singoli is not None) and (cluster_pod_singoli_nome is not None) and ((lista_pod_singoli is not None) or (regola_select is not None) or (regola_where is not None)):
            raise AttributeError('ERRORE: presente CLUSTER POD assieme a informazioni incompatibili')
        if (regola_select is not None) and ((lista_pod_singoli is not None) or (cluster_pod_singoli is not None) or (cluster_pod_singoli_nome is not None) or (regola_select is None)):
            raise AttributeError('ERRORE: presente REGOLA SELECT assieme a informazioni incompatibili')

        if lista_pod_singoli is not None:
            concat_syntax = f.concat_ws('#',
                                            f.coalesce(f.col('ZONA'), f.lit('')),
                                            f.coalesce(f.col('ID_AREA_GESTIONALE'), f.lit('')),
                                            f.coalesce(f.col('PROVINCIA_CRM'), f.lit('')),
                                            f.coalesce(f.col('TRATTAMENTO'), f.lit('')),
                                            f.coalesce(f.col('POD'), f.lit(''))
                                        )
            df_anagrafica = df_anagrafica.withColumn('CLUSTER', f.when(f.col('POD').isin(lista_pod_singoli), concat_syntax).otherwise(f.col('CLUSTER')))

        elif cluster_pod_singoli is not None:
            df_anagrafica = df_anagrafica.withColumn('CLUSTER', f.when(f.col('POD').isin(cluster_pod_singoli), f.lit(cluster_pod_singoli_nome)).otherwise(f.col('CLUSTER')))

        else:
            if not pd.isna(regola_where):
                df_anagrafica = df_anagrafica.filter(regola_where)
            df_anagrafica = df_anagrafica.withColumn('CLUSTER', f.expr(row['REGOLA_SELECT']))

    df_anagrafica = df_anagrafica.fillna('CLUSTER_MANCANTI', subset=['CLUSTER'])

    df_anagrafica = df_anagrafica.select(
        f.col('POD'),
        f.col('CLUSTER'),
        f.col('CONSUMO_ANNUO_COMPLESSIVO').alias('EAC')
    )

    ######################
    ### CALENDARIO-POD ###
    ######################

    df_calendario = spark.read.parquet(f's3://{NOME_BUCKET}/datamodel/CALENDARIO_GME') \
        .filter(f.col('DATA') == data_corrente) \
        .select(
            'UNIX_TIME',
            'TIMESTAMP',
            'DATA',
            'ORA_GME'
    )

    df_num_giorni_anno = spark.read.parquet(f's3://{NOME_BUCKET}/datamodel/CALENDARIO_GME') \
        .groupBy('ANNO').count().withColumnRenamed('count', 'NUM_GIORNO_ORA')

    df_best_choice = df_anagrafica.crossJoin(df_calendario)
    df_best_choice = df_best_choice.withColumn('ANNO', f.year(f.col('DATA')))
    df_best_choice = df_best_choice.join(df_num_giorni_anno, on=['ANNO'], how='left')
    df_best_choice = df_best_choice.withColumn('EAC_GIORNO_ORA', f.col('EAC')/f.col('NUM_GIORNO_ORA')) \
        .drop('NUM_GIORNO_ORA', 'ANNO')

    ##########################
    ### MISURE 1G - GIORNO ###
    ##########################

    df_1g_enelg = spark.sql(f"""
        SELECT * FROM 
            `623333656140/thr_prod_glue_db`.ee_curva_m1g AS CURVA
        WHERE 
            ymd = '{data_corrente.strftime("%Y/%m/%d")}' AND
            cd_flow = 'ENEL-G' AND
            grandezza = 'A'
    """)

    df_1g_enelg = df_1g_enelg.withColumn('MAX_TS', f.max('TS').over(Window.partitionBy('POD', 'YMD')))
    df_1g_enelg = df_1g_enelg.filter(f.col('TS') == f.col('MAX_TS')).drop('MAX_TS')

    #TODO: eliminare drop_duplicates
    df_1g_enelg = df_1g_enelg.dropDuplicates(subset=['POD', 'YMD', 'TS'])

    #if df_1g_enelg.groupBy('POD', 'YMD').count().filter(f.col('count')>1).count() > 0:
    #    check = df_1g_enelg.groupBy('POD', 'YMD').count().filter(f.col('count') > 1).select('POD', 'YMD')
    #    df_1g_enelg.join(check, on=['POD', 'YMD'], how='inner').repartition(1).write.mode('overwrite').csv(f's3://{NOME_BUCKET}/datamodel/DUPLICATI_1G')
    #    raise AttributeError('ERRORE: POD, GIORNO non chiave in 1g-giorno')

    for hh in range(0, 25):
        sum_col = f.col(f'con_e_{(hh*4+1):03}')
        for qh_idx in range(2, 5):
            sum_col = sum_col + f.col(f'con_e_{(hh*4+qh_idx):03}')
        df_1g_enelg = df_1g_enelg.withColumn(f'con_e_{(hh+1):02}', sum_col)

    df_1g_enelg = df_1g_enelg.select(
        f.col('POD'),
        f.to_date(f.col('ymd'), 'yyyy/MM/dd').alias('DATA'),
        *[f.col(f'con_e_{(hh+1):02}').cast(DoubleType()) for hh in range(0, 25)]
    )

    df_1g_enelg = melt(
        df=df_1g_enelg,
        id_vars=['POD', 'DATA'],
        value_vars=[f'con_e_{(hh+1):02}' for hh in range(0, 25)],
        var_name='ORA',
        value_name='ENERGIA_1G_GIORNO'
    )

    df_1g_enelg = df_1g_enelg.withColumn('ORA', f.substring(f.col('ORA'), -2, 2).cast(IntegerType()))
    df_1g_enelg = df_1g_enelg.withColumnRenamed('ORA', 'ORA_GME')

    df_best_choice = df_best_choice.join(df_1g_enelg, on=['POD', 'DATA', 'ORA_GME'], how='left')

    del df_1g_enelg

    ########################
    ### MISURE 1G - BEST ###
    ########################

    df_1g = spark.sql(f"""
        SELECT * FROM 
        `623333656140/thr_prod_glue_db`.ee_curva_m1g AS CURVA
        WHERE 
            ymd = '{data_corrente.strftime("%Y/%m/%d")}' AND
            grandezza = 'A'
    """)

    # df_misura = spark.sql('SELECT * FROM `623333656140/thr_prod_glue_db`.ee_misura_m1g') \
    #     .select(
    #         'id_misura_m2g',
    #         'motivazione'
    # )
    #
    # df_1g = df_1g.join(df_misura, on=['id_misura_m2g'], how='left')
    #
    # df_1g = df_1g.filter(f.col('motivazione') != '3')

    df_1g = df_1g.withColumn('RANK_ORIGINE',
        f.when(f.col('cd_flow')=='XLSX', f.lit(7))
         .when(f.col('cd_flow')=='RFO', f.lit(6))
         .when(((f.col('cd_flow')=='PDO') & (f.col('tipodato')=='E') & (f.col('validato')=='S')), f.lit(5))
         .when(((f.col('cd_flow')=='ENEL-M') | (f.col('cd_flow') == 'M15DL')), f.lit(4))
         .when(f.col('cd_flow')=='ENEL-G', f.lit(3))
         .when(f.col('cd_flow')=='SOS', f.lit(2))
         .when(((f.col('cd_flow')=='PDO') & (f.col('tipodato')!='E') & (f.col('validato')!='S')), f.lit(1))
         .otherwise(f.lit(-1))
    )

    w_1g = Window.partitionBy('POD', 'YMD').orderBy(f.col('RANK_ORIGINE').desc(), f.col('ts').desc())
    df_1g = df_1g.withColumn('RANK', f.row_number().over(w_1g))

    df_1g = df_1g.filter(f.col('RANK') == 1)

    #TODO: eliminare drop_duplicates -> eliminare quello con meno energia
    df_1g = df_1g.dropDuplicates(subset=['POD', 'YMD', 'TS'])

    #if df_1g.groupBy('POD', 'YMD').count().filter(f.col('count')>1).count() > 0:
    #    check = df_1g.groupBy('POD', 'YMD').count().filter(f.col('count') > 1).select('POD', 'YMD')
    #    df_1g.join(check, on=['POD', 'YMD'], how='inner').repartition(1).write.mode('overwrite').csv(f's3://{NOME_BUCKET}/datamodel/DUPLICATI_1G_BEST')
    #    raise AttributeError('ERRORE: POD, GIORNO non chiave in 1g')

    for hh in range(0, 25):
        sum_col = f.col(f'con_e_{(hh*4+1):03}')
        for qh_idx in range(2, 5):
            sum_col = sum_col + f.col(f'con_e_{(hh*4+qh_idx):03}')
        df_1g = df_1g.withColumn(f'con_e_{(hh+1):02}', sum_col)

    df_1g = df_1g.select(
        f.col('POD'),
        f.to_date(f.col('ymd'), 'yyyy/MM/dd').alias('DATA'),
        *[f.col(f'con_e_{(hh+1):02}').cast(DoubleType()) for hh in range(0, 25)]
    )

    df_1g = melt(
        df=df_1g,
        id_vars=['POD', 'DATA'],
        value_vars=[f'con_e_{(hh+1):02}' for hh in range(0, 25)],
        var_name='ORA',
        value_name='ENERGIA_1G_BEST'
    )

    df_1g = df_1g.withColumn('ORA', f.substring(f.col('ORA'), -2, 2).cast(IntegerType()))
    df_1g = df_1g.withColumnRenamed('ORA', 'ORA_GME')

    df_best_choice = df_best_choice.join(df_1g, on=['POD', 'DATA', 'ORA_GME'], how='left')

    del df_1g

    ########################
    ### MISURE 2G - BEST ###
    ########################

    df_2g = spark.sql(f"""
        SELECT * FROM 
        `623333656140/thr_prod_glue_db`.ee_curva_m2g_bc AS CURVA
        WHERE 
            ymd = '{data_corrente.strftime("%Y/%m/%d")}' AND
            grandezza = 'A' AND
            flg_best_choice = 'Y'
    """)

    df_2g = df_2g.withColumn('MAX_TS', f.max('TS').over(Window.partitionBy('POD', 'YMD')))
    df_2g = df_2g.filter(f.col('TS') == f.col('MAX_TS')).drop('MAX_TS')

    #TODO: eliminare drop_duplicates
    df_2g = df_2g.dropDuplicates(subset=['POD', 'YMD', 'TS'])

    #if df_2g.groupBy('POD', 'YMD').count().filter(f.col('count')>1).count() > 0:
    #    check = df_2g.groupBy('POD', 'YMD').count().filter(f.col('count') > 1).select('POD', 'YMD')
    #    df_2g.join(check, on=['POD', 'YMD'], how='inner').repartition(1).write.mode('overwrite').csv(f's3://{NOME_BUCKET}/datamodel/DUPLICATI_2G')
    #    raise AttributeError('ERRORE: POD, GIORNO non chiave in 2g')

    for hh in range(0, 25):
        sum_col = f.col(f'con_e_{(hh*4+1):03}')
        for qh_idx in range(2, 5):
            sum_col = sum_col + f.col(f'con_e_{(hh*4+qh_idx):03}')
        df_2g = df_2g.withColumn(f'con_e_{(hh+1):02}', sum_col)

    df_2g = df_2g.select(
        f.col('POD'),
        f.to_date(f.col('ymd'), 'yyyy/MM/dd').alias('DATA'),
        *[f.col(f'con_e_{(hh+1):02}').cast(DoubleType()) for hh in range(0, 25)]
    )

    df_2g = melt(
        df=df_2g,
        id_vars=['POD', 'DATA'],
        value_vars=[f'con_e_{(hh+1):02}' for hh in range(0, 25)],
        var_name='ORA',
        value_name='ENERGIA_2G_BEST'
    )

    df_2g = df_2g.withColumn('ORA', f.substring(f.col('ORA'), -2, 2).cast(IntegerType()))
    df_2g = df_2g.withColumnRenamed('ORA', 'ORA_GME')

    df_best_choice = df_best_choice.join(df_2g, on=['POD', 'DATA', 'ORA_GME'], how='left')

    del df_2g

    cond_1g = (f.col('ENERGIA_1G_GIORNO').isNotNull() | f.col('ENERGIA_1G_BEST').isNotNull()) & (f.col('ENERGIA_2G_BEST').isNull())
    cond_2g = (f.col('ENERGIA_1G_GIORNO').isNull() & f.col('ENERGIA_1G_BEST').isNull()) & (f.col('ENERGIA_2G_BEST').isNotNull())
    no_dato_cond = (f.col('ENERGIA_1G_GIORNO').isNull()) & (f.col('ENERGIA_1G_BEST').isNull()) & (f.col('ENERGIA_2G_BEST').isNull())

    df_best_choice = df_best_choice.withColumn('TIPOLOGIA_MISURA',
       f.when(cond_1g, f.lit('1G'))
       .when(cond_2g, f.lit('2G'))
       .when(no_dato_cond, f.lit('MANCANTE'))
       .otherwise(f.lit('MIX'))
    )

    #############################################
    ### SCRITTURA TABELLA ANALISI BEST-CHOICE ###
    #############################################

    if TRACE_ANALISI:
        outpath_trace = f's3://{NOME_BUCKET}/datamodel/{NOME_TABELLA}_ANALISI'
        delete_folder_s3(bucket=NOME_BUCKET, folder_path=f'datamodel/{NOME_TABELLA}_ANALISI/SIMULAZIONE={COD_SIMULAZIONE}/DATA={data_corrente.strftime("%Y-%m-%d")}/')
        df_best_choice.withColumn('SIMULAZIONE', f.lit(COD_SIMULAZIONE)).write.mode('append').partitionBy(['SIMULAZIONE', 'DATA']).parquet(outpath_trace)

    ############################
    ### AGGREGAZIONE CLUSTER ###
    ############################

    no_dato_cond = (f.col('ENERGIA_1G_GIORNO').isNull()) & (f.col('ENERGIA_1G_BEST').isNull()) & (f.col('ENERGIA_2G_BEST').isNull())

    df_best_choice = df_best_choice.groupBy('CLUSTER', 'DATA', 'UNIX_TIME', 'ORA_GME', 'TIMESTAMP').agg(
        f.sum('EAC_GIORNO_ORA').alias('EAC'),
        f.sum(f.when(no_dato_cond, f.col('EAC_GIORNO_ORA')).otherwise(f.lit(0.))).alias('EAC_MANCANTE'),
        f.count(f.lit(1)).alias('N_POD'),
        f.sum(f.when(f.col('TIPOLOGIA_MISURA')=='MANCANTE', f.lit(1)).otherwise(f.lit(0))).alias('N_POD_MANCANTI'),
        f.sum(f.when(f.col('TIPOLOGIA_MISURA')=='1G', f.lit(1)).otherwise(f.lit(0))).alias('N_MISURE_1G'),
        f.sum(f.when(f.col('TIPOLOGIA_MISURA')=='2G', f.lit(1)).otherwise(f.lit(0))).alias('N_MISURE_2G'),
        f.sum(f.when(f.col('TIPOLOGIA_MISURA')=='MIX', f.lit(1)).otherwise(f.lit(0))).alias('N_MISURE_MIX'),
        f.sum('ENERGIA_1G_GIORNO').alias('ENERGIA_1G_GIORNO'),
        f.sum('ENERGIA_1G_BEST').alias('ENERGIA_1G_BEST'),
        f.sum('ENERGIA_2G_BEST').alias('ENERGIA_2G_BEST')
    )

    df_best_choice = df_best_choice.withColumn('SIMULAZIONE', f.lit(COD_SIMULAZIONE))

    output_table = f's3://{NOME_BUCKET}/{tab_path}'
    delete_folder_s3(bucket=NOME_BUCKET, folder_path=f'datamodel/{NOME_TABELLA}/SIMULAZIONE={COD_SIMULAZIONE}/DATA={data_corrente.strftime("%Y-%m-%d")}/')
    df_best_choice.write.mode('append').partitionBy(['SIMULAZIONE', 'DATA']).parquet(output_table)

    data_corrente += relativedelta(days=1)
