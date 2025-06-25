import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# Es una buena práctica configurar el backend al principio del script.
# Si se ejecuta en un entorno sin GUI (como un servidor o notebook),
# 'Agg' es una mejor opción que 'TkAgg'.
# mpl.use('TkAgg') 

class WilliamsIndicators:
    """
    Calcula un conjunto de indicadores técnicos de Bill Williams sobre un DataFrame de velas.

    Esta clase toma un DataFrame de pandas con datos OHLCV y calcula:
    - Alligator (Mandíbula, Dientes, Labios)
    - Awesome Oscillator (AO)
    - Accelerator Oscillator (AC)
    - Barras Divergentes (Bullish/Bearish)
    - Ventanas de Profitunity (Basado en MFI)

    El método principal es `calculate_all()`, que procesa los datos y devuelve
    un DataFrame con todos los indicadores añadidos como nuevas columnas.
    """

    def __init__(self, candles_df: pd.DataFrame):
        """
        Inicializa la clase con el DataFrame de velas.

        Args:
            candles_df (pd.DataFrame): DataFrame que debe contener las columnas
                                      'Open', 'High', 'Low', 'Close', 'Volume'.
                                      Las columnas se esperan en minúsculas por convención.
        """
        # Hacemos una copia para no modificar el DataFrame original
        self.data = candles_df.copy()
        
        # Estandarizar nombres de columnas a minúsculas para consistencia
        self.data.columns = [col.lower() for col in self.data.columns]
        
        required_cols = {'open', 'high', 'low', 'close', 'volume'}
        if not required_cols.issubset(self.data.columns):
            raise ValueError(f"El DataFrame debe contener las columnas: {required_cols}")

    def _smma(self, series: pd.Series, period: int) -> pd.Series:
        """
        Calcula la Media Móvil Suavizada (Smoothed Moving Average - SMMA/RMA).
        Pandas ewm(alpha=1/period) es la implementación correcta y eficiente.
        """
        return series.ewm(alpha=1/period, adjust=False).mean()

    def _calculate_alligator(self):
        """Calcula las tres líneas del Alligator y las añade al DataFrame."""
        median_price = (self.data['high'] + self.data['low']) / 2
        
        # Mandíbula (Jaw) - Línea azul
        jaw = self._smma(median_price, 13).shift(8)
        
        # Dientes (Teeth) - Línea roja
        teeth = self._smma(median_price, 8).shift(5)
        
        # Labios (Lips) - Línea verde
        lips = self._smma(median_price, 5).shift(3)
        
        self.data['alligator_jaw'] = jaw
        self.data['alligator_teeth'] = teeth
        self.data['alligator_lips'] = lips

    def _calculate_ao_ac(self):
        """Calcula el Awesome Oscillator (AO) y el Accelerator Oscillator (AC)."""
        median_price = (self.data['high'] + self.data['low']) / 2
        
        # Awesome Oscillator (AO)
        sma5 = median_price.rolling(window=5).mean()
        sma34 = median_price.rolling(window=34).mean()
        self.data['awesome_oscillator'] = sma5 - sma34
        
        # Accelerator Oscillator (AC)
        sma5_ao = self.data['awesome_oscillator'].rolling(window=5).mean()
        self.data['accelerator_oscillator'] = self.data['awesome_oscillator'] - sma5_ao

    def _calculate_divergent_bars(self):
        """Identifica barras divergentes alcistas (bullish) y bajistas (bearish)."""
        prev_low = self.data['low'].shift(1)
        prev_high = self.data['high'].shift(1)
        
        # Barra divergente alcista (Bullish Divergent Bar)
        # Mínimo actual más bajo que el anterior, y el cierre está en la mitad superior de la barra.
        is_bull_divergent = (self.data['low'] < prev_low) & \
                            (self.data['close'] > (self.data['high'] + self.data['low']) / 2)
        self.data['is_bull_divergent_bar'] = is_bull_divergent

        # Barra divergente bajista (Bearish Divergent Bar)
        # Máximo actual más alto que el anterior, y el cierre está en la mitad inferior de la barra.
        is_bear_divergent = (self.data['high'] > prev_high) & \
                            (self.data['close'] < (self.data['high'] + self.data['low']) / 2)
        self.data['is_bear_divergent_bar'] = is_bear_divergent

    def _calculate_profitunity_windows(self):
        """
        Calcula las "ventanas de profitunity" basadas en el MFI y el volumen.
        - Green: MFI y Volumen suben.
        - Squat: MFI sube, Volumen baja.
        - Fade: MFI y Volumen bajan.
        - Fake: MFI baja, Volumen sube.
        """
        mfi = (self.data['high'] - self.data['low']) / self.data['volume']
        mfi = mfi.replace([np.inf, -np.inf], np.nan) # Evitar división por cero

        mfi_up = mfi > mfi.shift(1)
        volume_up = self.data['volume'] > self.data['volume'].shift(1)

        conditions = [
            mfi_up & volume_up,
            mfi_up & ~volume_up,
            ~mfi_up & ~volume_up,
            ~mfi_up & volume_up
        ]
        choices = ['green', 'squat', 'fade', 'fake']
        
        self.data['profitunity_window'] = np.select(conditions, choices, default=None)

    def calculate_all(self) -> pd.DataFrame:
        """
        Ejecuta todos los cálculos de indicadores y devuelve el DataFrame completo.

        Returns:
            pd.DataFrame: El DataFrame original con columnas adicionales para cada indicador.
        """
        if len(self.data) < 34:
            # 34 es el periodo más largo usado (para el AO)
            raise ValueError("Se necesitan al menos 34 periodos (filas) de datos.")
            
        print("Calculando Alligator...")
        self._calculate_alligator()
        
        print("Calculando Awesome Oscillator y Accelerator Oscillator...")
        self._calculate_ao_ac()
        
        print("Identificando Barras Divergentes...")
        self._calculate_divergent_bars()
        
        print("Calculando Ventanas de Profitunity...")
        self._calculate_profitunity_windows()
        
        print("Cálculos completados.")
        return self.data

    def plot_alligator(self, num_records: int = 100):
        """
        Grafica el indicador Alligator junto con el precio de cierre.

        Args:
            num_records (int): Número de los últimos registros a graficar.
        """
        subset = self.data.tail(num_records).copy()
        subset.reset_index(inplace=True) # Usar un índice numérico para el eje x

        plt.figure(figsize=(15, 7))
        plt.title(f'Bill Williams Alligator - Últimos {num_records} periodos')
        
        # Graficar precios
        plt.plot(subset['close'], label='Precio de Cierre', color='black', alpha=0.7)
        
        # Graficar Alligator
        plt.plot(subset['alligator_jaw'], label='Jaw (13, shift 8)', color='blue')
        plt.plot(subset['alligator_teeth'], label='Teeth (8, shift 5)', color='red')
        plt.plot(subset['alligator_lips'], label='Lips (5, shift 3)', color='green')
        
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.ylabel('Precio')
        plt.xlabel('Periodo')
        plt.show()

    def plot_oscillators(self, num_records: int = 100):
        """
        Grafica el Awesome Oscillator (AO) y el Accelerator Oscillator (AC).

        Args:
            num_records (int): Número de los últimos registros a graficar.
        """
        subset = self.data.tail(num_records).copy()
        subset.reset_index(inplace=True, drop=True)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
        fig.suptitle(f'Osciladores de Bill Williams - Últimos {num_records} periodos')

        # Awesome Oscillator (AO)
        colors_ao = ['green' if val >= 0 else 'red' for val in subset['awesome_oscillator']]
        ax1.bar(subset.index, subset['awesome_oscillator'], color=colors_ao, label='AO')
        ax1.axhline(0, color='grey', linestyle='--')
        ax1.set_title('Awesome Oscillator (AO)')
        ax1.grid(True, linestyle='--', alpha=0.5)
        ax1.legend()

        # Accelerator Oscillator (AC)
        colors_ac = ['green' if val >= 0 else 'red' for val in subset['accelerator_oscillator']]
        ax2.bar(subset.index, subset['accelerator_oscillator'], color=colors_ac, label='AC')
        ax2.axhline(0, color='grey', linestyle='--')
        ax2.set_title('Accelerator Oscillator (AC)')
        ax2.grid(True, linestyle='--', alpha=0.5)
        ax2.legend()
        
        plt.xlabel('Periodo')
        plt.show()


# --- Ejemplo de Uso ---
if __name__ == '__main__':
    # 1. Crear un DataFrame de ejemplo con datos de velas
    print("Creando datos de ejemplo...")
    np.random.seed(42)
    num_candles = 200
    prices = 100 + np.random.randn(num_candles).cumsum()
    sample_data = pd.DataFrame({
        'Open': prices + np.random.uniform(-0.5, 0.5, num_candles),
        'High': prices + np.random.uniform(0, 1, num_candles),
        'Low': prices - np.random.uniform(0, 1, num_candles),
        'Close': prices + np.random.uniform(-0.5, 0.5, num_candles),
        'Volume': np.random.randint(1000, 5000, num_candles)
    })
    # Asegurarse de que High sea el más alto y Low el más bajo
    sample_data['High'] = sample_data[['Open', 'High', 'Low', 'Close']].max(axis=1)
    sample_data['Low'] = sample_data[['Open', 'High', 'Low', 'Close']].min(axis=1)

    # 2. Instanciar la clase con los datos
    print("\nInicializando la clase WilliamsIndicators...")
    indicator_calculator = WilliamsIndicators(sample_data)

    # 3. Calcular todos los indicadores
    results_df = indicator_calculator.calculate_all()

    # 4. Mostrar los resultados
    print("\nDataFrame resultante con los indicadores (últimas 10 filas):")
    # Configurar pandas para mostrar todas las columnas
    pd.set_option('display.max_columns', None)
    print(results_df.tail(10))
    
    # 5. Graficar los indicadores
    print("\nGenerando gráficos...")
    indicator_calculator.plot_alligator(num_records=150)
    indicator_calculator.plot_oscillators(num_records=150)
