import boto3
import pandas as pd
from pyspark.sql import functions as f
from typing import Iterable
from pyspark.sql.dataframe import DataFrame as sparkDF
from pyspark.sql.functions import Column
from itertools import chain


def apply_map(
        df: sparkDF, col_name: str, map_dict: dict, na_replace=None,
        mantieni_mancanti: bool = True, default_mancanti=None,
        output_col_name: str = None, output_format=None
):
    """
    La funzione consente di decodificare una colonna di un dataframe SPARK sulla base delle informazioni
    contenute nel dizionario map_dict, vi sono parametri obbligatori e parametri facoltativi

    PARAMETRI OBBLIGATORI:
        - df: dataframe SPARK a cui applicare la decodifica
        - col_name: nome della colonna di df da decodificare
        - map_dict: mappa con i valori di decodifica {val1_old:val1_new, val2_old:val2_new}
            il valore val1_old verrà sostituito con val1_new e così via
    PARAMETRI FACOLTATIVI:
        - na_replace: il valore con cui vengono sostituiti gli elementi nulli della colonna col_name
            se non valorizzato restituiscono NA
        - mantieni_mancanti: se non valorizzato i valori mancanti nel dizionario vengono lasciati invariati
            i campi non mappati in map_dict altrimenti se True vengono valorizzati in base al campo default_mancanti
        - default_mancanti: valore a cui sono valorizzati i valori mancanti nel dizionario map_dict se
            mantieni_mancanti == True
        - output_col_name: se non valorizzato la colonna originale viene sovrascritta altrimenti se il campo è valorizzato
            crea una colonna con le decodifiche
        - output_format: se non valorizzato il formato della colonna dopo la decodifica rimane immutato altrimenti
            viene cambiato sulla base di questo valore, accetta sia tipi nativi di Spark che il formato stringa
            ('string','int','double' ... )
    """
    if not isinstance(col_name, Column):
        col_name = f.col(col_name)
    mapping_expr = f.create_map([f.lit(x) for x in chain(*map_dict.items())])
    if pd.isna(output_col_name):
        output_col_name = col_name._jc.toString()
    if mantieni_mancanti:
        expr = f.when(~f.isnull(mapping_expr[col_name]), mapping_expr[col_name]).otherwise(col_name)
    else:
        expr = f.when(~f.isnull(mapping_expr[col_name]), mapping_expr[col_name]).otherwise(f.lit(default_mancanti))
    if not pd.isna(output_format):
        expr = expr.cast(output_format)
    df = df.withColumn(output_col_name, expr)
    if na_replace is not None:
        df = df.fillna(na_replace, subset=[output_col_name])
    return df


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

