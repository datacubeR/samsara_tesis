import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class AnalyzeResample:
    """ Clase para Entregar un Análisis Completo 
    de los Nulos de las Series de Tiempo.
    """    
    def __init__(self, df, rule):
        """
        Parameters
        ----------
        df : DataFrame
            DataFrame con las series de Tiempo.
        rule : Str
            Offset String para el Resample.
        """        
        self.df = df
        self.rule = rule

    def resample(self):
        """Genera el Resample a probar generando las mismas variables del dataset Original.
        """
        variables = ['date', 'ndvi', 'ts_id', 'ndvi_null', 'id']
        self.df_resample = (self.df.groupby('ts_id')
            .resample(on = 'date', rule = self.rule).mean()
            .reset_index()
            .assign(ndvi_null = lambda x: x.ndvi.isnull(),
                    id = lambda x: x.ts_id + '-' + x.groupby('ts_id').date.rank().astype('int').astype(str)))[variables]
        

    def plot_nulls(self):
        """Genera una comparación entre los Nulos resultantes entre el Resample y el Original.
        """        
        plt.figure(figsize = (10,8))
        self.df.groupby('ts_id').ndvi_null.sum().plot(label = 'Nulos Normales')
        self.df_resample.groupby('ts_id').ndvi_null.sum().plot(label = 'Nulos Resample')
        plt.title(f'Comparación con Resample {self.rule}')
        plt.legend()
        plt.show();
    
    def hist_nulls(self):
        """Genera un Histograma de los Nulos del Resample y el Original.
        """        
        (self.df.groupby('ts_id').ndvi_null.sum()
            .plot(kind = 'hist', figsize = (10,8), bins = 30, label = 'Histograma Nulos', alpha = 0.5))

        (self.df_resample.groupby('ts_id').ndvi_null.sum()
            .plot(kind = 'hist', figsize = (10,8), bins = 30, label = 'Histograma Nulos Resample', alpha = 0.5))
        
        plt.title('Histograma de Cantidad de Nulos por TS')
        plt.legend()
        plt.show();
        
    def analyze(self):
        self.resample()
        self.plot_nulls()
        self.hist_nulls()
        return self.df_resample


def import_file(file, id_length = 4):
    """Importa todos los archivos CSV de prueba convirtiendolo en un sólo DataFrame.

    Parameters
    ----------
    file : str
        Path que apunta a alguno de los archivos CSV a utilizar.
    id_length : int, optional
        Número de Caracteres para utilizar en el Identificador de la serie de tiempo, por defecto 4.

    Returns
    -------
    DataFrame
        DataFrame con todos las series de tiempo.
    """    
    ts_id = file.split('_')[-1].split('.')[0][-id_length:]
    output = pd.read_csv(file, index_col=0, parse_dates = ['date']).assign(ts_id = ts_id)
    return output


def plot_examples(df, threshold):
    """Genera Ejemplos de Series de Tiempo con un Cierto Threshold de Nulos.

    Parameters
    ----------
    df : DataFrame
        DataFrame con Series de Tiempo a mostrar.
    threshold : int
        Se Mostrarán las series de tiempo que tengan más nulos que el threshold.
    """    
    idxs = df.groupby('ts_id').ndvi_null.sum().loc[lambda x: x > threshold]
    for index, value in zip(idxs.index, idxs):
        df.set_index('date').query('ts_id == @index').plot(figsize = (10,8), title = f'TS = [{index}] con {value} Nulos')


def ts_stats(df):
    """Cálculo de Estadísticas de interés.

    Parameters
    ----------
    df : DataFrame
        DataFrame con Series de Tiempo a mostrar.
    """    
    largo_prom = df.groupby("ts_id").ndvi.size().mean()
    nulos = df.groupby('ts_id').ndvi_null.sum().mean()
    print(f'Número de TSs: {df.ts_id.nunique()}')
    print(f'Largo Promedio por TS: {largo_prom}')
    print(f'Promedio de Nulos por TS: {nulos}')
    
    
def show_original_vs_resamples(df_real, df_resamples, k, figsize = (30,40)):
    """Función para comparar k series de tiempos provenientes de atasets, 
    normalmente el original versus el Resampleado. 
    
    Parameters
    ----------
    df_real : DataFrame
        DataFrame con la data sin resamplear.
    df_resamples : DataFrame
        DataFrame con la data resampleada.
    k : int
        Número de Series de Tiempo a mostrar.
    figsize : tuple, optional
        Tamaño del Gráfico a mostrar, by default (30,40)
    """    
    ids = df_resamples.ts_id.unique()
    ids_selected = random.choices(ids, k = k)

    fig, axes = plt.subplots(nrows=10, ncols=2, figsize=figsize)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.1, hspace=1)

    for n, id in enumerate(ids_selected):
        df_real.set_index('date').query('ts_id == @id').plot(title = f'Ejemplos TS = {id} Original', ax = axes[n,0])
        df_resamples.set_index('date').query('ts_id == @id').plot(title = f'Ejemplos TS = {id} con Resample',ax = axes[n,1])


def show_k_sequences(sequences, dates, k = 10, figsize = (10,20)):
    """Grafica K Secuencias aleatorias para ver su comportamiento.

    Parameters
    ----------
    sequences : Numpy Array
        Secuencias a graficar con Forma (N_seq x Seq_Len)
    dates : Numpy Array
        Fechas asociadas a las secuencias con Forma (N_seq x Seq_Len)
    k : int, optional
        Número de Secuencias a graficar, by default 10
    figsize : tuple, optional
        Tamaño del Gráfico, by default (10,20)
    """    
    VALUES = np.random.randint(0,len(sequences),k)

    plt.figure(figsize = figsize)
    for k, id in enumerate(VALUES, start = 1):
        plt.subplot(10,1,k)
        plt.plot(dates[id], sequences[id])

def create_sequences(df, seq_len):
    sequences = []
    divisor = []
    indices = []
    dates = []
    target = []
    dates_target = []
    target_idx = []
    ids = df.ts_id.unique()
    for id in ids:
        ts = df.query('ts_id == @id').to_numpy()

        subsequences = np.zeros((len(ts)-seq_len+1, seq_len))
        subindices = np.empty((len(ts)-seq_len+1, seq_len), dtype = 'object')
        subdates = np.empty((len(ts)-seq_len+1, seq_len), dtype = 'datetime64[D]')
        subdivisor = np.zeros(len(ts))
        
        for i in range(len(ts)-seq_len+1):
            subsequences[i] = ts[i:i+seq_len,1]
            subindices[i] = ts[i:i+seq_len,4]
            subdates[i] = ts[i:i+seq_len,0]
            subdivisor[i:i+seq_len] += np.ones(seq_len)
            
            
            target.append(ts[i+seq_len,1] if i != len(ts)-seq_len else np.nan )
            target_idx.append(ts[i+seq_len,4] if i != len(ts)-seq_len else np.nan )
            dates_target.append(ts[i+seq_len,0] if i != len(ts)-seq_len else np.nan )
            
        sequences.extend(subsequences)
        divisor.append(subdivisor)
        indices.extend(subindices)
        dates.extend(subdates)
        

    print(f'Se crearon {len(sequences)} secuencias y {len(target)} targets.')
        
    return (np.array(sequences), divisor, 
            np.array(indices), np.array(dates), 
            np.array(target), np.array(dates_target),
            np.array(target_idx))
