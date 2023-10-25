import sys
import subprocess

subprocess.check_call([sys.executable, "-m", "pip", "install", "boto3"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas"])

from datetime import datetime
from pyspark.sql.types import *
from pyspark.sql import functions as f
from pyspark.sql import SparkSession
from typing import Iterable
from pyspark.sql.dataframe import DataFrame as sparkDF
import boto3
from dateutil.relativedelta import relativedelta
import pandas as pd
from pyspark.sql import Window
import logging

COD_SIMULAZIONE = sys.argv[1]
NOME_TABELLA = 'BEST_CHOICE_NEW'
NOME_BUCKET = 'eva-qa-s3-model'
TRACE_ANALISI = eval(sys.argv[2])

INIZIO_PERIODO = datetime.strptime(sys.argv[3], '%Y-%m-%d').date()
FINE_PERIODO = datetime.strptime(sys.argv[4], '%Y-%m-%d').date()
tab_path = f'datamodel/{NOME_TABELLA}/'

########################
### GESTIONE DEI LOG ###
########################


def check_path_s3(path: str, is_file: bool = False):
    if len(path) > 0:
        if path[-1] != '/' and not is_file: path = f'{path}/'
        if path[0] == '/': path = path[1:]
    return path


class LogStream(object):
    def __init__(self):
        self.logs = ''

    def write(self, log_str):
        self.logs += log_str

    def flush(self):
        pass

    def __str__(self):
        return self.logs

    def __repr__(self):
        return self.logs


class S3logger(object):
    def __init__(self, bucket_name: str, folder_path: str, nome_log: str = ''):
        self.logger = logging.getLogger(nome_log)
        self.log_stream = LogStream()
        self.bucket_name = bucket_name
        self.key = f'{check_path_s3(folder_path)}{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}.log'
        log_handler = logging.StreamHandler(self.log_stream)
        log_handler.setLevel(logging.INFO)
        log_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s:\n%(message)s\n'))
        self.logger.addHandler(log_handler)
        self.logger.info('Inizio logging')

    def logs3(self, log_message: str, level: str = 'INFO'):
        level = level.upper()
        if level == 'DEBUG':
            print('loggo debug')
            self.logger.debug(log_message)
        elif level == 'INFO':
            print('loggo info')
            self.logger.info(log_message)
        elif level == 'WARNING':
            print('loggo warning')
            self.logger.warning(log_message)
        elif level == 'ERROR':
            print('loggo error')
            self.logger.error(log_message)
        elif level == 'ERROR':
            print('loggo error')
            self.logger.critical(log_message)
        else:
            self.logger.error(f'Livello {level} non definito\n{log_message}')
        boto3.client("s3").put_object(Body=self.log_stream.logs, Bucket=self.bucket_name, Key=self.key)


# logger = logging.getLogger('BEST CHOICE')
#
# log_stream = LogStream(bucket_name=NOME_BUCKET, folder_path=f'logs/{NOME_TABELLA}/')
# log_handler = logging.StreamHandler(log_stream)
# log_handler.setLevel(logging.INFO)
# log_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s:\n%(message)s\n'))
#
# logger.addHandler(log_handler)
#
# logger.debug('debugging something')
# logger.info('informing user')
#
# boto3.client("s3").put_object(Body=log_stream.logs, Bucket=log_stream.BUCKET_NAME, Key=log_stream.KEY)

log = S3logger(bucket_name=NOME_BUCKET, folder_path=f'logs/{NOME_TABELLA}', nome_log=NOME_TABELLA)

log.logs3('test')

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

log.info("Hello S3 - NUMERO 2")

spark = SparkSession.builder.appName('EVA - BEST CHOICE').enableHiveSupport().getOrCreate()

data_corrente = INIZIO_PERIODO

##############################
### CONFIGURAZIONE CLUSTER ###
##############################

df_conf_cluster = spark.read.parquet(f's3://{NOME_BUCKET}/datamodel/CONFIGURAZIONE_CLUSTER')

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

    df_anagrafica = df_anagrafica.filter(f.col('TIPO_MISURATORE').isin(['E', 'G', 'O']))

    df_anagrafica = df_anagrafica.withColumn('DT_INI_VALIDITA', f.to_timestamp(f.col('DT_INI_VALIDITA'), 'dd/MM/yyyy HH:mm:SS'))
    df_anagrafica = df_anagrafica.withColumn('DT_FIN_VALIDITA', f.to_timestamp(f.col('DT_FIN_VALIDITA'), 'dd/MM/yyyy HH:mm:SS'))
    df_anagrafica = df_anagrafica.withColumn('CONSUMO_ANNUO_COMPLESSIVO', f.regexp_replace(f.col('CONSUMO_ANNUO_COMPLESSIVO'), ',', '.').cast(DoubleType()))

    # Filtriamo solo i pod che hanno inizio e fine fornitura nella data corrente
    df_anagrafica = df_anagrafica.filter(
        (f.col('DT_FIN_VALIDITA') >= (data_corrente + relativedelta(days=1))) &
        (f.col('DT_INI_VALIDITA') <= (data_corrente + relativedelta(days=1)))
    )

    # CODIZIONI PER CAPIRE SE POD E' 1G OPPURE 2G
    cond_flusso_dati = f.when(f.col('TIPO_MISURATORE') == 'G', f.lit('2G')).when(f.col('TIPO_MISURATORE').isin(['E', 'O']), f.lit('1G')).otherwise(f.lit('ERRORE'))
    df_anagrafica = df_anagrafica.withColumn('FLUSSO_DATI', cond_flusso_dati)

    #TODO: gestione degli errori in anagrafica

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
        f.col('FLUSSO_DATI'),
        f.col('CONSUMO_ANNUO_COMPLESSIVO').alias('EAC')
    )

    #TODO: eliminare drop duplicates: non dovrebbero esistere pod doppi
    df_anagrafica = df_anagrafica.dropDuplicates(subset=['POD'])

    ######################
    ### CALENDARIO-POD ###
    ######################

    df_calendario = spark.read.parquet(f's3://{NOME_BUCKET}/datamodel/CALENDARIO_GME') \
        .filter(f.col('DATA') == data_corrente) \
        .select(
            'UNIX_TIME',
            'TIMESTAMP',
            'DATA',
            'ORA_GME',
            'GIORNO_SETTIMANA'
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
    df_1g_enelg = df_1g_enelg.withColumn('FLUSSO_DATI', f.lit('1G'))

    df_best_choice = df_best_choice.join(df_1g_enelg, on=['POD', 'DATA', 'ORA_GME', 'FLUSSO_DATI'], how='left')

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
    df_1g = df_1g.withColumn('FLUSSO_DATI', f.lit('1G'))

    df_best_choice = df_best_choice.join(df_1g, on=['POD', 'DATA', 'ORA_GME', 'FLUSSO_DATI'], how='left')

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
    df_2g = df_2g.withColumn('FLUSSO_DATI', f.lit('2G'))

    df_best_choice = df_best_choice.join(df_2g, on=['POD', 'DATA', 'ORA_GME', 'FLUSSO_DATI'], how='left')

    del df_2g

    ####################################
    ### AGGIUSTIAMO I CAMPI MANCANTI ###
    ####################################

    # Recuperiamo i valori mancanti prendendo l'ultimo dato disponibile corrispondente al primo giorno della settimana mancante
    w_recupero_na = Window.partitionBy('GIORNO_SETTIMANA').orderBy('UNIX_TIME').rowsBetween(Window.unboundedPreceding, Window.currentRow)
    no_dato_cond = (f.col('ENERGIA_1G_GIORNO').isNull()) & (f.col('ENERGIA_1G_BEST').isNull()) & (f.col('ENERGIA_2G_BEST').isNull())

    df_best_choice = df_best_choice.withColumn('ENERGIA_1G_BEST_RECUPERO_NA', f.first(f.col('ENERGIA_1G_BEST'), ignorenulls=True).over(w_recupero_na))
    df_best_choice = df_best_choice.withColumn('ENERGIA_2G_BEST_RECUPERO_NA', f.first(f.col('ENERGIA_2G_BEST'), ignorenulls=True).over(w_recupero_na))
    df_best_choice = df_best_choice.withColumn('FLAG_DATO_MANCANTE', f.when(no_dato_cond, f.lit('Y')).otherwise(f.lit('N')))

    df_best_choice = df_best_choice.withColumn('ENERGIA_1G_MISURA', f.coalesce(f.col('ENERGIA_1G_BEST'), f.col('ENERGIA_1G_GIORNO')))
    df_best_choice = df_best_choice.withColumn('ENERGIA_2G_MISURA', f.col('ENERGIA_2G_BEST'))

    df_best_choice = df_best_choice.withColumn('ENERGIA_1G_MANCANTE',
        f.when(
           f.col('ENERGIA_1G_MISURA').isNull(),
           f.col('ENERGIA_1G_BEST_RECUPERO_NA')
        ).otherwise(
           f.lit(None).cast(DoubleType())
        )
    )
    df_best_choice = df_best_choice.withColumn('ENERGIA_2G_MANCANTE',
        f.when(
           f.col('ENERGIA_2G_MISURA').isNull(),
           f.col('ENERGIA_2G_BEST_RECUPERO_NA')
        ).otherwise(
           f.lit(None).cast(DoubleType())
        )
    )

    df_best_choice = df_best_choice.withColumn('CONSUMI_MISURA',
        f.when(f.col('FLUSSO_DATI') == '1G', f.col('ENERGIA_1G_MISURA'))
         .when(f.col('FLUSSO_DATI') == '2G', f.col('ENERGIA_1G_MISURA'))
    )

    df_best_choice = df_best_choice.withColumn('CONSUMI_MANCANTI',
        f.when(f.col('FLUSSO_DATI') == '1G', f.col('ENERGIA_1G_MANCANTE'))
         .when(f.col('FLUSSO_DATI') == '2G', f.col('ENERGIA_2G_MANCANTE'))
    )

    df_best_choice = df_best_choice.withColumn('CONSUMI', f.coalesce(f.col('CONSUMI_MISURA'), f.col('CONSUMI_MANCANTI'), f.col('EAC_GIORNO_ORA')))

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

    df_best_choice = df_best_choice.groupBy('CLUSTER', 'DATA', 'UNIX_TIME', 'ORA_GME', 'TIMESTAMP', 'FLUSSO_DATI').agg(
        f.count(f.lit(1)).alias('N_POD'),
        f.sum(f.when(f.col('FLAG_DATO_MANCANTE') == 'Y', f.lit(1)).otherwise(f.lit(0))).alias('N_POD_MANCANTI'),
        f.sum('CONSUMI').alias('CONSUMI'),
        f.sum('CONSUMI_MANCANTI').alias('CONSUMI_MANCANTI')
    )

    df_best_choice = df_best_choice.withColumn('SIMULAZIONE', f.lit(COD_SIMULAZIONE))

    output_table = f's3://{NOME_BUCKET}/{tab_path}'
    delete_folder_s3(bucket=NOME_BUCKET, folder_path=f'datamodel/{NOME_TABELLA}/SIMULAZIONE={COD_SIMULAZIONE}/DATA={data_corrente.strftime("%Y-%m-%d")}/')
    df_best_choice.write.mode('append').partitionBy(['SIMULAZIONE', 'DATA']).parquet(output_table)

    data_corrente += relativedelta(days=1)
