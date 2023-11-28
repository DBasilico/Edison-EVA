import boto3
import pandas as pd
from datetime import date
from pyspark.sql.types import *
from pyspark.sql import functions as f
from pyspark.sql import Window
from typing import Iterable
from pyspark.sql.dataframe import DataFrame as sparkDF
from dateutil.relativedelta import relativedelta
from datetime import datetime

def get_subfolders_s3(bucket: str, path: str = ''):
    path = check_path_s3(path)
    response = boto3.client('s3').list_objects_v2(Bucket=bucket, Prefix=path, Delimiter='/')
    if 'CommonPrefixes' not in response.keys():
        return
    else:
        for obj in response["CommonPrefixes"]:
            subfolder = obj['Prefix']
            yield subfolder
            yield from get_subfolders_s3(bucket, path=subfolder)


def output_path_recupero_na(nome_bucket: str, IS_PREVISIVO: bool, INIZIO_PERIODO: date) -> str:
    if IS_PREVISIVO:
        nome_tabella = f'BEST_CHOICE_RECUPERO_NA_PREVISIVO_{INIZIO_PERIODO.strftime("%Y%m%d")}'
    else:
        nome_tabella = f'BEST_CHOICE_RECUPERO_NA_CONSUNTIVO_{INIZIO_PERIODO.strftime("%Y%m%d")}'
    return f's3://{nome_bucket}/datamodel/{nome_tabella}'


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
    cols = id_vars + [f.col("_vars_and_vals")[x].alias(x) for x in [var_name, value_name]]
    return _tmp.select(*cols)


def extract_1g_giorno(
    spark,
    INIZIO_PERIODO: date,
    FINE_PERIODO: date,
    IS_PREVISIONE: bool,
    DATA_BACKTEST: datetime
) -> sparkDF:

    df_1g_enelg = spark.sql(f"""
        SELECT * FROM 
            `623333656140/thr_prod_glue_db`.ee_curva_m1g AS CURVA
        WHERE
            cd_flow = 'ENEL-G' AND
            grandezza = 'A'
    """)

    if IS_PREVISIONE:
        df_1g_enelg = df_1g_enelg.filter(f.col('tms_pubblicazione') < DATA_BACKTEST - relativedelta(days=1))

    df_1g_enelg = df_1g_enelg.withColumn('DATA', f.to_date(f.col('ymd'), 'yyyy/MM/dd'))
    df_1g_enelg = df_1g_enelg.filter(f.col('DATA').between(INIZIO_PERIODO, FINE_PERIODO))

    df_1g_enelg = df_1g_enelg.withColumn('MAX_TS', f.max('TS').over(Window.partitionBy('POD', 'DATA')))
    df_1g_enelg = df_1g_enelg.filter(f.col('TS') == f.col('MAX_TS')).drop('MAX_TS')

    # TODO: eliminare drop_duplicates
    df_1g_enelg = df_1g_enelg.dropDuplicates(subset=['POD', 'DATA', 'TS'])

    for hh in range(0, 25):
        sum_col = f.col(f'con_e_{(hh * 4 + 1):03}')
        for qh_idx in range(2, 5):
            sum_col = sum_col + f.col(f'con_e_{(hh * 4 + qh_idx):03}')
        df_1g_enelg = df_1g_enelg.withColumn(f'con_e_{(hh + 1):02}', sum_col)

    df_1g_enelg = df_1g_enelg.select(
        f.col('POD'),
        f.col('DATA'),
        *[f.col(f'con_e_{(hh + 1):02}').cast(DoubleType()) for hh in range(0, 25)]
    )

    df_1g_enelg = melt(
        df=df_1g_enelg,
        id_vars=['POD', 'DATA'],
        value_vars=[f'con_e_{(hh + 1):02}' for hh in range(0, 25)],
        var_name='ORA',
        value_name='ENERGIA_1G_GIORNO'
    )

    df_1g_enelg = df_1g_enelg.withColumn('ORA', f.substring(f.col('ORA'), -2, 2).cast(IntegerType()))
    df_1g_enelg = df_1g_enelg.withColumnRenamed('ORA', 'ORA_GME')

    return df_1g_enelg


def extract_1g_best(spark, INIZIO_PERIODO: date, FINE_PERIODO: date, IS_PREVISIONE: bool, DATA_BACKTEST: datetime) -> sparkDF:

    df_1g = spark.sql(f"""
        SELECT * FROM 
        `623333656140/thr_prod_glue_db`.ee_curva_m1g AS CURVA
        WHERE 
            grandezza = 'A'
    """)

    if IS_PREVISIONE:
        df_1g = df_1g.filter(f.col('tms_pubblicazione') < DATA_BACKTEST - relativedelta(days=1))

    df_1g = df_1g.withColumn('DATA', f.to_date(f.col('ymd'), 'yyyy/MM/dd'))
    df_1g = df_1g.filter(f.col('DATA').between(INIZIO_PERIODO, FINE_PERIODO))

    df_1g = df_1g.withColumn('RANK_ORIGINE',
        f.when(f.col('cd_flow') == 'XLSX', f.lit(7))
        .when(f.col('cd_flow') == 'RFO', f.lit(6))
        .when(((f.col('cd_flow') == 'PDO') & (f.col('tipodato') == 'E') & (f.col('validato') == 'S')), f.lit(5))
        .when(((f.col('cd_flow') == 'ENEL-M') | (f.col('cd_flow') == 'M15DL')), f.lit(4))
        .when(f.col('cd_flow') == 'ENEL-G', f.lit(3))
        .when(f.col('cd_flow') == 'SOS', f.lit(2))
        .when(((f.col('cd_flow') == 'PDO') & (f.col('tipodato') != 'E') & (f.col('validato') != 'S')), f.lit(1))
        .otherwise(f.lit(-1))
    )

    w_1g = Window.partitionBy('POD', 'DATA').orderBy(f.col('RANK_ORIGINE').desc(), f.col('TS').desc())
    df_1g = df_1g.withColumn('RANK', f.row_number().over(w_1g))

    df_1g = df_1g.filter(f.col('RANK') == 1)

    # TODO: eliminare drop_duplicates -> eliminare quello con meno energia
    df_1g = df_1g.dropDuplicates(subset=['POD', 'DATA', 'TS'])

    for hh in range(0, 25):
        sum_col = f.col(f'con_e_{(hh * 4 + 1):03}')
        for qh_idx in range(2, 5):
            sum_col = sum_col + f.col(f'con_e_{(hh * 4 + qh_idx):03}')
        df_1g = df_1g.withColumn(f'con_e_{(hh + 1):02}', sum_col)

    df_1g = df_1g.select(
        f.col('POD'),
        f.col('DATA'),
        *[f.col(f'con_e_{(hh + 1):02}').cast(DoubleType()) for hh in range(0, 25)]
    )

    df_1g = melt(
        df=df_1g,
        id_vars=['POD', 'DATA'],
        value_vars=[f'con_e_{(hh + 1):02}' for hh in range(0, 25)],
        var_name='ORA',
        value_name='ENERGIA_1G_BEST'
    )

    df_1g = df_1g.withColumn('ORA', f.substring(f.col('ORA'), -2, 2).cast(IntegerType()))
    df_1g = df_1g.withColumnRenamed('ORA', 'ORA_GME')

    return df_1g


def extract_2g_best(spark, INIZIO_PERIODO: date, FINE_PERIODO: date, IS_PREVISIONE: bool, DATA_BACKTEST: datetime) -> sparkDF:

    df_2g = spark.sql(f"""
        SELECT * FROM 
        `623333656140/thr_prod_glue_db`.ee_curva_m2g_bc AS CURVA
        WHERE 
            grandezza = 'A' AND
            flg_best_choice = 'Y'
    """)

    if IS_PREVISIONE:
        df_2g = df_2g.filter(f.col('tms_pubblicazione') < DATA_BACKTEST - relativedelta(days=1))

    df_2g = df_2g.withColumn('DATA', f.to_date(f.col('ymd'), 'yyyy/MM/dd'))
    df_2g = df_2g.filter(f.col('DATA').between(INIZIO_PERIODO, FINE_PERIODO))

    df_2g = df_2g.withColumn('MAX_TS', f.max('TS').over(Window.partitionBy('POD', 'DATA')))
    df_2g = df_2g.filter(f.col('TS') == f.col('MAX_TS')).drop('MAX_TS')

    # TODO: eliminare drop_duplicates
    df_2g = df_2g.dropDuplicates(subset=['POD', 'DATA', 'TS'])

    for hh in range(0, 25):
        sum_col = f.col(f'con_e_{(hh * 4 + 1):03}')
        for qh_idx in range(2, 5):
            sum_col = sum_col + f.col(f'con_e_{(hh * 4 + qh_idx):03}')
        df_2g = df_2g.withColumn(f'con_e_{(hh + 1):02}', sum_col)

    df_2g = df_2g.select(
        f.col('POD'),
        f.col('DATA'),
        *[f.col(f'con_e_{(hh + 1):02}').cast(DoubleType()) for hh in range(0, 25)]
    )

    df_2g = melt(
        df=df_2g,
        id_vars=['POD', 'DATA'],
        value_vars=[f'con_e_{(hh + 1):02}' for hh in range(0, 25)],
        var_name='ORA',
        value_name='ENERGIA_2G_BEST'
    )

    df_2g = df_2g.withColumn('ORA', f.substring(f.col('ORA'), -2, 2).cast(IntegerType()))
    df_2g = df_2g.withColumnRenamed('ORA', 'ORA_GME')

    return df_2g


def build_recupero_na(
        spark,
        NOME_BUCKET: str,
        DATA_INIZIO: date,
        DATA_FINE: date,
        GIORNI_INDIETRO_RECUPERO: int,
        IS_PREVISIONE: bool,
        DATA_BACKTEST: datetime
) -> str:

    wnd_recupero = Window.partitionBy('POD', 'ORA_GME', 'GIORNO_SETTIMANA').orderBy(f.col('DATA').desc())

    ### CALENDARIO ###

    df_calendario = spark.read.parquet(f's3://{NOME_BUCKET}/datamodel/CALENDARIO_GME') \
        .filter(f.col('DATA').between(DATA_INIZIO - relativedelta(days=1 + 60), DATA_FINE)) \
        .select(
            'DATA',
            'ORA_GME',
            'GIORNO_SETTIMANA'
    )

    ### MISURE 1G - BEST ###
    df_1g = extract_1g_best(spark, DATA_INIZIO - relativedelta(days=1 + 60), DATA_FINE, IS_PREVISIONE, DATA_BACKTEST)
    df_1g = df_1g.withColumnRenamed('ENERGIA_1G_BEST','ENERGIA_1G_RECUPERO_NA')
    df_1g = df_1g.join(df_calendario, on=['DATA', 'ORA_GME'], how='left').dropna(subset=['ENERGIA_1G_RECUPERO_NA'])
    df_1g = df_1g.withColumn('RANK', f.row_number().over(wnd_recupero)).filter(f.col('RANK') == 1).drop('RANK')
    df_1g = df_1g.select('POD', 'ORA_GME', 'GIORNO_SETTIMANA', 'ENERGIA_1G_RECUPERO_NA')

    ### MISURE 2G - BEST ###
    df_2g = extract_2g_best(spark, DATA_INIZIO - relativedelta(days=1 + 15), DATA_FINE, IS_PREVISIONE, DATA_BACKTEST)
    df_2g = df_2g.withColumnRenamed('ENERGIA_2G_BEST', 'ENERGIA_2G_RECUPERO_NA')
    df_2g = df_2g.join(df_calendario, on=['DATA', 'ORA_GME'], how='left').dropna(subset=['ENERGIA_2G_RECUPERO_NA'])
    df_2g = df_2g.withColumn('RANK', f.row_number().over(wnd_recupero)).filter(f.col('RANK') == 1).drop('RANK')
    df_2g = df_2g.select('POD', 'ORA_GME', 'GIORNO_SETTIMANA', 'ENERGIA_2G_RECUPERO_NA')

    ### RECUPERO NA ###
    df_recupero_na = df_1g.join(df_2g, on=['POD', 'ORA_GME', 'GIORNO_SETTIMANA'], how='outer')

    output_path = output_path_recupero_na(nome_bucket=NOME_BUCKET, IS_PREVISIVO=IS_PREVISIONE, INIZIO_PERIODO=DATA_INIZIO)

    df_recupero_na.repartition(50, 'POD').write.mode('overwrite').partitionBy('GIORNO_SETTIMANA').parquet(output_path)

    return output_path


def get_conf_cluster(spark, nome_tabella_configurazione: str, nome_bucket: str, cod_simulazione: str) -> sparkDF:
    df_conf_cluster = spark.read.parquet(f's3://{nome_bucket}/datamodel/{nome_tabella_configurazione}')
    df_conf_cluster = df_conf_cluster.filter(f.col('SIMULAZIONE') == cod_simulazione)
    max_validita = df_conf_cluster.agg(f.max('VALIDITA').alias('MAX_VALIDITA')).collect()[0]['MAX_VALIDITA']
    df_conf_cluster = df_conf_cluster.filter(f.col('VALIDITA') == max_validita)
    df_conf_cluster = df_conf_cluster.pandas_api().set_index('ORDINAMENTO')
    df_conf_cluster.sort_index(inplace=True)
    if not df_conf_cluster.index.is_unique:
        raise AttributeError('Presente ORDINAMENTO con valori multipli')
    return df_conf_cluster


def get_anagrafica(
        spark,
        nome_tabella_anagrafica: str,
        nome_bucket: str,
        inizio_periodo: date,
        fine_periodo: date
) -> sparkDF:
    # POD attivi in cui dobbiamo calcolare la BEST CHOICE (DATA FORNITURA)

    folder_anagrafica = get_subfolders_s3(bucket=nome_bucket, path=nome_tabella_anagrafica)
    folder_anagrafica = max([x.split('/')[-2] for x in folder_anagrafica])

    df_anagrafica = spark.read.parquet(f's3://{nome_bucket}/{nome_tabella_anagrafica}/{folder_anagrafica}')
    df_anagrafica = df_anagrafica.filter(f.col('TRATTAMENTO') == 'O')
    df_anagrafica = df_anagrafica.withColumn('TIPO_FLUSSO',
                                             f.when(f.col('TIPO_MISURATORE') == 'G', f.lit('2G')).otherwise(
                                                 f.lit('1G')))
    df_anagrafica = df_anagrafica.withColumn('DT_INI_VALIDITA', f.to_timestamp(f.col('DT_INIZIO_VALIDITA'), 'yyyy-MM-dd'))
    df_anagrafica = df_anagrafica.withColumn('DT_FIN_VALIDITA', f.to_timestamp(f.col('DT_FINE_VALIDITA'), 'yyyy-MM-dd'))
    df_anagrafica = df_anagrafica.withColumn('DT_INI_VALIDITA', f.to_date(f.date_trunc('day', f.col('DT_INI_VALIDITA'))))
    df_anagrafica = df_anagrafica.withColumn('DT_FIN_VALIDITA', f.to_date(f.date_trunc('day', f.col('DT_FIN_VALIDITA'))))

    df_anagrafica = df_anagrafica.filter(
        (f.col('DT_INI_VALIDITA') <= fine_periodo) & (f.col('DT_FIN_VALIDITA') >= inizio_periodo)
    )

    # df_anagrafica = df_anagrafica.withColumn('CONSUMO_ANNUO_COMPLESSIVO',
    #                                          f.regexp_replace(f.col('CONSUMO_ANNUO_COMPLESSIVO'), ',', '.').cast(
    #                                              DoubleType()))
    # df_anagrafica = df_anagrafica.filter(
    #     ~((f.col('DT_FIN_VALIDITA') < inizio_periodo) | (f.col('DT_INI_VALIDITA') > fine_periodo))
    # )

    return df_anagrafica


def build_cluster(df_anagrafica: sparkDF, df_conf_cluster: sparkDF) -> sparkDF:
    df_anagrafica = df_anagrafica.withColumn('CLUSTER', f.lit(None).cast(StringType()))

    for k_rule in df_conf_cluster.index.to_numpy():
        row = df_conf_cluster.loc[int(k_rule)]
        lista_pod_singoli = row['LISTA_POD_SINGOLI']
        cluster_pod_singoli = row['CLUSTER_POD_SINGOLI']
        cluster_pod_singoli_nome = row['CLUSTER_POD_SINGOLI_NOME']
        regola_select = row['REGOLA_SELECT']
        regola_where = row['REGOLA_WHERE']
        if (lista_pod_singoli is not None) and (
                (cluster_pod_singoli is not None) or (cluster_pod_singoli_nome is not None) or (
                regola_select is not None) or (regola_where is not None)):
            raise AttributeError('ERRORE: presente LISTA POD SINGOLI assieme a informazioni incompatibili')
        if (cluster_pod_singoli is None) and (cluster_pod_singoli_nome is not None):
            raise AttributeError('ERRORE: presente NOME CLUSTER POD SINGOLI ma manca LISTA')
        if (cluster_pod_singoli is not None) and (cluster_pod_singoli_nome is None):
            raise AttributeError('ERRORE: presente CLUSTER POD SINGOLI ma manca NOME')
        if (cluster_pod_singoli is not None) and (cluster_pod_singoli_nome is not None) and (
                (lista_pod_singoli is not None) or (regola_select is not None) or (regola_where is not None)):
            raise AttributeError('ERRORE: presente CLUSTER POD assieme a informazioni incompatibili')
        if (regola_select is not None) and ((lista_pod_singoli is not None) or (cluster_pod_singoli is not None) or (
                cluster_pod_singoli_nome is not None) or (regola_select is None)):
            raise AttributeError('ERRORE: presente REGOLA SELECT assieme a informazioni incompatibili')
        if lista_pod_singoli is not None:
            concat_syntax = f.concat_ws(
                '#',
                f.coalesce(f.col('ZONA'), f.lit('')),
                f.coalesce(f.col('ID_AREA_GESTIONALE'), f.lit('')),
                f.lit(''),
                f.lit(''),
                f.coalesce(f.col('POD'), f.lit(''))
            )
            df_anagrafica = df_anagrafica.withColumn('CLUSTER', f.when(f.col('POD').isin(lista_pod_singoli),
                                                                       concat_syntax).otherwise(f.col('CLUSTER')))
        elif cluster_pod_singoli is not None:
            df_anagrafica = df_anagrafica.withColumn('CLUSTER', f.when(f.col('POD').isin(cluster_pod_singoli),
                                                                       f.lit(cluster_pod_singoli_nome)).otherwise(
                f.col('CLUSTER')))
        else:
            if not pd.isna(regola_where):
                df_anagrafica = df_anagrafica.withColumn('CLUSTER', f.expr(
                    f'CASE WHEN {regola_where} THEN {regola_select} ELSE CLUSTER END'))
            else:
                df_anagrafica = df_anagrafica.withColumn('CLUSTER', f.expr(regola_select))

    df_anagrafica = df_anagrafica.withColumn('CLUSTER', f.coalesce(f.col('CLUSTER'), f.lit('CLUSTER_MANCANTI')))

    df_anagrafica = df_anagrafica.select(
        f.col('POD'),
        f.col('CLUSTER'),
        f.col('DT_INI_VALIDITA'),
        f.col('DT_FIN_VALIDITA'),
        f.col('CONSUMO_ANNUO')
    )

    # TODO: eliminare drop duplicates: non dovrebbero esistere pod doppi
    return df_anagrafica.dropDuplicates(subset=['POD'])


def add_period(
        spark,
        nome_bucket: str,
        nome_tabella_calendario: str,
        df_anagrafica: sparkDF,
        inizio_periodo: date,
        fine_periodo: date
) -> sparkDF:
    df_calendario = spark.read.parquet(f's3://{nome_bucket}/datamodel/{nome_tabella_calendario}') \
        .filter(f.col('DATA').between(inizio_periodo, fine_periodo)) \
        .select(
        'UNIX_TIME',
        'TIMESTAMP',
        'DATA',
        'ORA_GME',
        'GIORNO_SETTIMANA'
    )

    df_best_choice = df_anagrafica.crossJoin(df_calendario)

    df_best_choice = df_best_choice.filter(f.col('DATA').between(f.col('DT_INI_VALIDITA'), f.col('DT_FIN_VALIDITA')))

    df_num_giorni_anno = spark.read.parquet(f's3://{nome_bucket}/datamodel/{nome_tabella_calendario}') \
        .groupBy('ANNO').count().withColumnRenamed('count', 'NUM_GIORNO_ORA')
    df_best_choice = df_best_choice.withColumn('ANNO', f.year(f.col('DATA')))
    df_best_choice = df_best_choice.join(df_num_giorni_anno, on=['ANNO'], how='left')
    df_best_choice = df_best_choice.withColumn('EAC', f.col('CONSUMO_ANNUO') / f.col('NUM_GIORNO_ORA')) \
        .drop('NUM_GIORNO_ORA', 'ANNO')
    return df_best_choice


def add_consumi(
    spark,
    nome_bucket: str,
    effettuare_recupero: bool,
    inizio_periodo: date,
    fine_periodo: date,
    giorni_indietro_recupero: int,
    df_best_choice: sparkDF,
    IS_PREVISIONE: bool,
    DATA_BACKTEST: datetime
) -> sparkDF:

    ### MISURE 1G - GIORNO ###
    df_1g_giorno = extract_1g_giorno(spark, inizio_periodo, fine_periodo, IS_PREVISIONE, DATA_BACKTEST)
    df_best_choice = df_best_choice.join(df_1g_giorno, on=['POD', 'DATA', 'ORA_GME'], how='left')
    del df_1g_giorno

    ### MISURE 1G - BEST ###
    df_1g_best = extract_1g_best(spark, inizio_periodo, fine_periodo, IS_PREVISIONE, DATA_BACKTEST)
    df_best_choice = df_best_choice.join(df_1g_best, on=['POD', 'DATA', 'ORA_GME'], how='left')
    del df_1g_best

    ### MISURE 2G - BEST ###
    df_2g_best = extract_2g_best(spark, inizio_periodo, fine_periodo, IS_PREVISIONE, DATA_BACKTEST)
    df_best_choice = df_best_choice.join(df_2g_best, on=['POD', 'DATA', 'ORA_GME'], how='left')
    del df_2g_best

    ### RECUPERO DATI MANCANTI ###
    if effettuare_recupero:
        df_recupero_na = spark.read.parquet(
            build_recupero_na(spark, nome_bucket, inizio_periodo, fine_periodo, giorni_indietro_recupero, IS_PREVISIONE, DATA_BACKTEST)
        )
        df_best_choice = df_best_choice.join(df_recupero_na, on=['POD', 'ORA_GME', 'GIORNO_SETTIMANA'], how='left')
    else:
        df_best_choice = df_best_choice.withColumn('ENERGIA_1G_RECUPERO_NA', f.lit(None).cast(DoubleType()))
        df_best_choice = df_best_choice.withColumn('ENERGIA_2G_RECUPERO_NA', f.lit(None).cast(DoubleType()))

    df_best_choice = df_best_choice.withColumn('IS_MISSING', f.when(
        (f.col('ENERGIA_1G_BEST').isNull()) & (f.col('ENERGIA_1G_GIORNO').isNull()) & (
            f.col('ENERGIA_2G_BEST').isNull()),
        f.lit(1)
    ).otherwise(
        f.lit(0)
    ))

    df_best_choice = df_best_choice.withColumn('ENERGIA_1G_BEST_VALORE', f.when(
        f.col('ENERGIA_1G_BEST').isNotNull(),
        f.col('ENERGIA_1G_BEST')
    ).otherwise(
        f.lit(None).cast(DoubleType())
    ))

    df_best_choice = df_best_choice.withColumn('ENERGIA_1G_GIORNO_VALORE', f.when(
        (f.col('ENERGIA_1G_BEST').isNull()) & (f.col('ENERGIA_1G_GIORNO').isNotNull()),
        f.col('ENERGIA_1G_GIORNO')
    ).otherwise(
        f.lit(None).cast(DoubleType())
    ))

    df_best_choice = df_best_choice.withColumn('ENERGIA_2G_BEST_VALORE', f.when(
        (f.col('ENERGIA_1G_BEST').isNull()) & (f.col('ENERGIA_1G_GIORNO').isNull()) & (
            f.col('ENERGIA_2G_BEST').isNotNull()),
        f.col('ENERGIA_2G_BEST')
    ).otherwise(
        f.lit(None).cast(DoubleType())
    ))

    df_best_choice = df_best_choice.withColumn('ENERGIA_NA_VALORE', f.when(
        f.col('IS_MISSING') == 1,
        f.coalesce(f.col('ENERGIA_1G_RECUPERO_NA'), f.col('ENERGIA_2G_RECUPERO_NA'))
    ).otherwise(
        f.lit(None)
    ))

    df_best_choice = df_best_choice.withColumn('ENERGIA_EAC_VALORE', f.when(
        (f.col('IS_MISSING') == 1) & (f.col('ENERGIA_NA_VALORE').isNull()),
        f.col('EAC')
    ).otherwise(
        f.lit(None)
    ))

    df_best_choice = df_best_choice.withColumn('CONSUMI', f.coalesce(
        f.col('ENERGIA_1G_BEST'),
        f.col('ENERGIA_1G_GIORNO'),
        f.col('ENERGIA_2G_BEST'),
        f.col('ENERGIA_1G_RECUPERO_NA'),
        f.col('ENERGIA_2G_RECUPERO_NA'),
        f.col('EAC')
    ))

    return df_best_choice


def delete_recupero_na(nome_bucket: str, IS_PREVISIVO: bool, DATA_INIZIO:date):
    df_recupero_na = output_path_recupero_na(nome_bucket=nome_bucket, IS_PREVISIVO=IS_PREVISIVO, INIZIO_PERIODO=DATA_INIZIO)
    cutting_point = df_recupero_na.find(nome_bucket) + len(nome_bucket) + 1
    df_recupero_na = df_recupero_na[cutting_point:]
    delete_folder_s3(bucket=nome_bucket, folder_path=df_recupero_na)

