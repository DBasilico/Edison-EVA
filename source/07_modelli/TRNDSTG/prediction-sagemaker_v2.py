import random

import numpy as np
import pandas as pd
from pandas.core.nanops import nanmean as pd_nanmean
from statsmodels.tools.validation import PandasWrapper, array_like
from statsmodels.tsa.filters.filtertools import convolution_filter
from statsmodels.tsa.tsatools import freq_to_period
from pmdarima.arima import auto_arima


# Define the function to return the MAPE values
def mape(actual, predicted) -> float:
    # Convert actual and predicted
    # to numpy array data type if not already
    if not all([isinstance(actual, np.ndarray),
                isinstance(predicted, np.ndarray)]):
        actual, predicted = np.array(actual),
        np.array(predicted)

    # Calculate the MAPE value and return
    return round(np.mean(np.abs((actual - predicted) / actual)) * 100, 5)

def _extrapolate_trend(trend, npoints):
    """
    Replace nan values on trend's end-points with least-squares extrapolated
    values with regression considering npoints closest defined points.
    """
    front = next(
        i for i, vals in enumerate(trend) if not np.any(np.isnan(vals))
    )
    back = (
        trend.shape[0]
        - 1
        - next(
            i
            for i, vals in enumerate(trend[::-1])
            if not np.any(np.isnan(vals))
        )
    )
    front_last = min(front + npoints, back)
    back_first = max(front, back - npoints)

    k, n = np.linalg.lstsq(
        np.c_[np.arange(front, front_last), np.ones(front_last - front)],
        trend[front:front_last],
        rcond=-1,
    )[0]
    extra = (np.arange(0, front) * np.c_[k] + np.c_[n]).T
    if trend.ndim == 1:
        extra = extra.squeeze()
    trend[:front] = extra

    k, n = np.linalg.lstsq(
        np.c_[np.arange(back_first, back), np.ones(back - back_first)],
        trend[back_first:back],
        rcond=-1,
    )[0]
    extra = (np.arange(back + 1, trend.shape[0]) * np.c_[k] + np.c_[n]).T
    if trend.ndim == 1:
        extra = extra.squeeze()
    trend[back + 1 :] = extra

    return trend


def seasonal_mean(x, period):
    """
    Return means for each period in x. period is an int that gives the
    number of periods per cycle. E.g., 12 for monthly. NaNs are ignored
    in the mean.
    """
    return np.array([pd_nanmean(x[i::period], axis=0) for i in range(period)])


def seasonal_decompose(
    x,
    model="additive",
    filt=None,
    period=None,
    two_sided=True,
    extrapolate_trend=0,
):
    """
    Seasonal decomposition using moving averages.

    Parameters
    ----------
    x : array_like
        Time series. If 2d, individual series are in columns. x must contain 2
        complete cycles.
    model : {"additive", "multiplicative"}, optional
        Type of seasonal component. Abbreviations are accepted.
    filt : array_like, optional
        The filter coefficients for filtering out the seasonal component.
        The concrete moving average method used in filtering is determined by
        two_sided.
    period : int, optional
        Period of the series. Must be used if x is not a pandas object or if
        the index of x does not have  a frequency. Overrides default
        periodicity of x if x is a pandas object with a timeseries index.
    two_sided : bool, optional
        The moving average method used in filtering.
        If True (default), a centered moving average is computed using the
        filt. If False, the filter coefficients are for past values only.
    extrapolate_trend : int or 'freq', optional
        If set to > 0, the trend resulting from the convolution is
        linear least-squares extrapolated on both ends (or the single one
        if two_sided is False) considering this many (+1) closest points.
        If set to 'freq', use `freq` closest points. Setting this parameter
        results in no NaN values in trend or resid components.

    Returns
    -------
    DecomposeResult
        A object with seasonal, trend, and resid attributes.

    See Also
    --------
    statsmodels.tsa.filters.bk_filter.bkfilter
        Baxter-King filter.
    statsmodels.tsa.filters.cf_filter.cffilter
        Christiano-Fitzgerald asymmetric, random walk filter.
    statsmodels.tsa.filters.hp_filter.hpfilter
        Hodrick-Prescott filter.
    statsmodels.tsa.filters.convolution_filter
        Linear filtering via convolution.
    statsmodels.tsa.seasonal.STL
        Season-Trend decomposition using LOESS.

    Notes
    -----
    This is a naive decomposition. More sophisticated methods should
    be preferred.

    The additive model is Y[t] = T[t] + S[t] + e[t]

    The multiplicative model is Y[t] = T[t] * S[t] * e[t]

    The results are obtained by first estimating the trend by applying
    a convolution filter to the data. The trend is then removed from the
    series and the average of this de-trended series for each period is
    the returned seasonal component.
    """
    pfreq = period
    pw = PandasWrapper(x)
    if period is None:
        pfreq = getattr(getattr(x, "index", None), "inferred_freq", None)

    x = array_like(x, "x", maxdim=2)
    nobs = len(x)

    if not np.all(np.isfinite(x)):
        raise ValueError("This function does not handle missing values")
    if model.startswith("m"):
        if np.any(x <= 0):
            raise ValueError(
                "Multiplicative seasonality is not appropriate "
                "for zero and negative values"
            )

    if period is None:
        if pfreq is not None:
            pfreq = freq_to_period(pfreq)
            period = pfreq
        else:
            raise ValueError(
                "You must specify a period or x must be a pandas object with "
                "a PeriodIndex or a DatetimeIndex with a freq not set to None"
            )
    if x.shape[0] < 2 * pfreq:
        raise ValueError(
            f"x must have 2 complete cycles requires {2 * pfreq} "
            f"observations. x only has {x.shape[0]} observation(s)"
        )

    if filt is None:
        if period % 2 == 0:  # split weights at ends
            filt = np.repeat(1.0 / period, period)
        else:
            filt = np.repeat(1.0 / period, period)

    nsides = int(two_sided) + 1
    trend = convolution_filter(x, filt, nsides)

    if extrapolate_trend == "freq":
        extrapolate_trend = period - 1

    if extrapolate_trend > 0:
        trend = _extrapolate_trend(trend, extrapolate_trend + 1)

    if model.startswith("m"):
        detrended = x / trend
    else:
        detrended = x - trend

    period_averages = seasonal_mean(detrended, period)

    if model.startswith("m"):
        period_averages /= np.mean(period_averages, axis=0)
    else:
        period_averages -= np.mean(period_averages, axis=0)

    seasonal = np.tile(period_averages.T, nobs // period + 1).T[:nobs]

    if model.startswith("m"):
        resid = x / seasonal / trend
    else:
        resid = detrended - seasonal

    results = []
    for s, name in zip(
        (seasonal, trend, resid, x), ("seasonal", "trend", "resid", None)
    ):
        results.append(pw.wrap(s.squeeze(), columns=name))
    return DecomposeResult(
        seasonal=results[0],
        trend=results[1],
        resid=results[2],
        observed=results[3],
    )


class DecomposeResult:
    """
    Results class for seasonal decompositions

    Parameters
    ----------
    observed : array_like
        The data series that has been decomposed.
    seasonal : array_like
        The seasonal component of the data series.
    trend : array_like
        The trend component of the data series.
    resid : array_like
        The residual component of the data series.
    weights : array_like, optional
        The weights used to reduce outlier influence.
    """

    def __init__(self, observed, seasonal, trend, resid, weights=None):
        self._seasonal = seasonal
        self._trend = trend
        if weights is None:
            weights = np.ones_like(observed)
            if isinstance(observed, pd.Series):
                weights = pd.Series(
                    weights, index=observed.index, name="weights"
                )
        self._weights = weights
        self._resid = resid
        self._observed = observed

    @property
    def observed(self):
        """Observed data"""
        return self._observed

    @property
    def seasonal(self):
        """The estimated seasonal component"""
        return self._seasonal

    @property
    def trend(self):
        """The estimated trend component"""
        return self._trend

    @property
    def resid(self):
        """The estimated residuals"""
        return self._resid

    @property
    def weights(self):
        """The weights used in the robust estimation"""
        return self._weights

    @property
    def nobs(self):
        """Number of observations"""
        return self._observed.shape

    def plot(
            self,
            observed=True,
            seasonal=True,
            trend=True,
            resid=True,
            weights=False,
        ):
            """
            Plot estimated components

            Parameters
            ----------
            observed : bool
                Include the observed series in the plot
            seasonal : bool
                Include the seasonal component in the plot
            trend : bool
                Include the trend component in the plot
            resid : bool
                Include the residual in the plot
            weights : bool
                Include the weights in the plot (if any)

            Returns
            -------
            matplotlib.figure.Figure
                The figure instance that containing the plot.
            """
            from pandas.plotting import register_matplotlib_converters

            from statsmodels.graphics.utils import _import_mpl

            plt = _import_mpl()
            register_matplotlib_converters()
            series = [(self._observed, "Observed")] if observed else []
            series += [(self.trend, "trend")] if trend else []

            if self.seasonal.ndim == 1:
                series += [(self.seasonal, "seasonal")] if seasonal else []
            elif self.seasonal.ndim > 1:
                if isinstance(self.seasonal, pd.DataFrame):
                    for col in self.seasonal.columns:
                        series += (
                            [(self.seasonal[col], "seasonal")] if seasonal else []
                        )
                else:
                    for i in range(self.seasonal.shape[1]):
                        series += (
                            [(self.seasonal[:, i], "seasonal")] if seasonal else []
                        )

            series += [(self.resid, "residual")] if resid else []
            series += [(self.weights, "weights")] if weights else []

            if isinstance(self._observed, (pd.DataFrame, pd.Series)):
                nobs = self._observed.shape[0]
                xlim = self._observed.index[0], self._observed.index[nobs - 1]
            else:
                xlim = (0, self._observed.shape[0] - 1)

            fig, axs = plt.subplots(len(series), 1)
            for i, (ax, (series, def_name)) in enumerate(zip(axs, series)):
                if def_name != "residual":
                    ax.plot(series)
                else:
                    ax.plot(series, marker="o", linestyle="none")
                    ax.plot(xlim, (0, 0), color="#000000", zorder=-3)
                name = getattr(series, "name", def_name)
                if def_name != "Observed":
                    name = name.capitalize()
                title = ax.set_title if i == 0 and observed else ax.set_ylabel
                title(name)
                ax.set_xlim(xlim)

            fig.tight_layout()
            return fig

import os
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import joblib
import io
from urllib.parse import unquote
import shutil
from datetime import datetime
from urllib.parse import quote
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
import os
import pathlib
from dateutil import tz

ROME_TZ = tz.gettz('Europe/Rome')

def datetime_to_epoch(ser):
    if ser.hasnans:
        res = ser.dropna().astype('int64').astype('Int64').reindex(index=ser.index)
    else:
        res = ser.astype('int64')
    return res // 10**9


def predict_fn(input_data, model):
    df_out = pd.DataFrame()

    CLUSTER_LIST = model.CLUSTER.unique()
    model = model.drop_duplicates(subset=['CLUSTER']).set_index('CLUSTER')

    # input_data = input_data[input_data.UNIX_TIME >= datetime(2023, 1, 1, tzinfo=ROME_TZ).timestamp()].copy()

    data_previsione = datetime(2023, 9, 18, tzinfo=ROME_TZ)

    try:

        while data_previsione < datetime(2023, 9, 30, tzinfo=ROME_TZ):

            print(data_previsione)

            df_data = input_data[input_data.UNIX_TIME < (data_previsione+relativedelta(days=1)).timestamp()].copy()

            for CLUSTER in CLUSTER_LIST:
                df = df_data[df_data.CLUSTER == CLUSTER].copy()

                df_params_cluster = model.loc[CLUSTER]

                df['CONSUMI_1G'] = df['ENERGIA_1G_BEST'].combine_first(df['ENERGIA_1G_GIORNO']).fillna(0.).replace(
                    np.nan, 0.)
                df['CONSUMI_2G'] = df['ENERGIA_2G_BEST'].fillna(0.).replace(np.nan, 0.)
                df['CONSUMI'] = df['CONSUMI_1G'] + df['CONSUMI_2G']

                df = df[df.ORA_GME < 25]

                df['CONSUMI_NORMALIZZATI'] = np.where(df['N_POD'] == df['N_POD_MANCANTI'], df['CONSUMI'] / df['N_POD'],
                                                      df['CONSUMI'] / (df['N_POD'] - df['N_POD_MANCANTI']))

                df = df[['UNIX_TIME', 'TIMESTAMP', 'CONSUMI_NORMALIZZATI', 'N_POD', 'N_POD_MANCANTI']]
                df = df.rename(columns={'TIMESTAMP': 'TIMESTAMP_STRING', 'CONSUMI_NORMALIZZATI': 'CONSUMI'})

                df = df.astype(
                    {'UNIX_TIME': np.int64, 'CONSUMI': np.float64, 'N_POD': np.int32, 'N_POD_MANCANTI': np.int32})
                df['TIMESTAMP'] = pd.to_datetime(df['UNIX_TIME'], unit='s', utc=True).dt.tz_convert('Europe/Rome')
                df = df.set_index('TIMESTAMP').sort_index()

                if df_params_cluster['TIPO_MODELLO'] == 'multiplicative' and (df['CONSUMI'] <= 0).any():
                    print('---> ADEGUAMENTO MODELLO - ADDITTIVO <---')
                    df_params_cluster['TIPO_MODELLO'] = 'addittive'

                #df_calcolo = df[df.index < data_previsione]

                seasonal_dec_result = seasonal_decompose(df['CONSUMI'], model=df_params_cluster['TIPO_MODELLO'],
                                                         two_sided=False, period=df_params_cluster['PERIODO'])
                trend = seasonal_dec_result.trend
                seasonal = seasonal_dec_result.seasonal

                #max_trend_day = max(trend.index).to_pydatetime() - relativedelta(days=1)
                #for pred_iter in range(24):
                    #mean = (trend.loc[max_trend_day + relativedelta(hours=pred_iter)] + trend.loc[max_trend_day + relativedelta(hours=pred_iter - 1)]) / 2
                    #trend.loc[max_trend_day + relativedelta(hours=pred_iter + 1)] = mean
                    #seasonal.loc[max_trend_day + relativedelta(hours=pred_iter + 1)] = seasonal.loc[max_trend_day + relativedelta(hours=pred_iter + 1 - df_params_cluster['PERIODO'])]

                if random.uniform(0,1) > 0.5:
                    print('ARIMA')
                    trend_arima = trend[
                        (trend.index < data_previsione) & (trend.index >= datetime(2023, 9, 1, tzinfo=ROME_TZ))]
                    trend_pred = auto_arima(trend_arima).predict(n_periods=24)
                    prediction = trend_pred + seasonal[seasonal.index >= data_previsione]
                else:
                    prediction = trend+seasonal
                    prediction = prediction[prediction.index >= data_previsione]

                prediction.name = 'PREVISIONE'
                prediction = prediction.to_frame()

                prediction = prediction.merge(df[['CONSUMI', 'N_POD', 'N_POD_MANCANTI']],  how='left', left_index=True, right_index=True)
                prediction['UNIX_TIME'] = datetime_to_epoch(prediction.index)

                prediction['PREVISIONE'] = np.where(prediction['N_POD'] == prediction['N_POD_MANCANTI'],
                                                    prediction['PREVISIONE'] * prediction['N_POD'],
                                                    prediction['PREVISIONE'] * (
                                                                prediction['N_POD'] - prediction['N_POD_MANCANTI']))
                prediction['CONSUMI'] = np.where(prediction['N_POD'] == prediction['N_POD_MANCANTI'],
                                                 prediction['CONSUMI'] * prediction['N_POD'],
                                                 prediction['CONSUMI'] * (
                                                             prediction['N_POD'] - prediction['N_POD_MANCANTI']))

                prediction = prediction[['PREVISIONE', 'CONSUMI', 'UNIX_TIME']]
                prediction['CLUSTER'] = CLUSTER
                prediction['COD_SIMULAZIONE'] = '1000'
                prediction['COD_MODELLO'] = 'TEST_SAGEMAKER'
                prediction['GIORNI_PREVISIONE'] = 3
                prediction['WMAE'] = np.sum(np.abs(prediction['CONSUMI'].values - prediction['PREVISIONE'].values))/np.sum(prediction['CONSUMI'].values)
                prediction['ZONA'] = CLUSTER.split('#')[0]

                df_out = pd.concat([df_out, prediction])

                #output_wmae = prediction.reset_index(names='TIMESTAMP')[['ZONA', 'TIMESTAMP', 'CONSUMI', 'PREVISIONE']].groupby(['ZONA', 'TIMESTAMP']).sum()
                #for ZONA in output_wmae.index.get_level_values('ZONA').unique():
                #    print(CLUSTER)
                #    test = output_wmae.loc[ZONA]
                #    print(np.sum(np.abs(test['CONSUMI'].values - test['PREVISIONE'].values)) / np.sum(test['CONSUMI'].values))
                #    print(np.sum(test['CONSUMI'].values))

            data_previsione = data_previsione + relativedelta(days=1)

    except Exception as e:
        print('ERRORE')
        print(e)
        raise Exception('ERRORE')

    return df_out



import joblib
model = joblib.load('MODELDATA/model_train_eva.joblib')
input_data = pd.read_parquet('MODELDATA/data.parquet')
input_data = input_data[input_data.UNIX_TIME >= datetime(2023, 1, 1, tzinfo=ROME_TZ).timestamp()]

output = predict_fn(input_data, model)

output.to_parquet('output_2.parquet')

output['ZONA'] = [x.split('#')[0] for x in output['CLUSTER']]
output.reset_index(inplace=True, names='TIMESTAMP')
output_wmae = output[['ZONA', 'TIMESTAMP', 'CONSUMI', 'PREVISIONE']].groupby(['ZONA', 'TIMESTAMP']).sum()

for ZONA in output_wmae.index.get_level_values('ZONA').unique():
    print(ZONA)
    test = output_wmae.loc[ZONA]
    print(np.sum(np.abs(test['CONSUMI'].values - test['PREVISIONE'].values))/np.sum(test['CONSUMI'].values))

