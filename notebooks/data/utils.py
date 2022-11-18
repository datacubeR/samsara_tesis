import glob
import pandas as pd
import matplotlib.pyplot as plt


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


class ImportSeries:
    def __init__(self, path: str, suffix: str, id_length: int = 4):
        self.path = path
        self.id_length = id_length
        self.suffix = suffix

    @staticmethod
    def import_file(file, id_length, suffix):
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
        output = pd.read_csv(file, index_col=0, parse_dates = ['date']).assign(ts_id = ts_id + suffix)
        return output

    def to_df(self):
        files = glob.glob(self.path)
        df = pd.concat([self.import_file(file, self.id_length, self.suffix) for file in files]).assign(ndvi_null = lambda x: x.ndvi.isnull())
        return df


class AnalyzeResample:
    """ Clase para Entregar un Análisis Completo 
    de los Nulos de las Series de Tiempo.
    """    
    def __init__(self, df: pd.DataFrame, rule: str):
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
        self.df.groupby('ts_id').ndvi_null.sum().plot(label = 'Nulos sin Resample', kind = "bar")
        self.df_resample.groupby('ts_id').ndvi_null.sum().plot(label = 'Nulos Resample', color = "orange")
        plt.xticks(rotation = 90)
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

    def plot_examples(self, threshold, n = 10):
        """Genera Ejemplos de Series de Tiempo con al menos Cierto Threshold de Nulos.

        Parameters
        ----------
        df : DataFrame
            DataFrame con Series de Tiempo a mostrar.
        threshold : int
            Se Mostrarán las series de tiempo que tengan más nulos que el threshold.
        """    

        idxs = self.df.groupby('ts_id').ndvi_null.sum().loc[lambda x: x > threshold].sample(n, random_state = 42)
        for index, value in zip(idxs.index, idxs):
            self.df.set_index('date').query('ts_id == @index').plot(figsize = (10,8), title = f'TS = [{index}] con {value} Nulos')
