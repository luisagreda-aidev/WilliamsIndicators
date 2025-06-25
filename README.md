# Bill Williams Technical Indicators Calculator

Una clase de Python eficiente y fácil de usar para calcular un conjunto de indicadores técnicos de trading desarrollados por el legendario trader Bill Williams. La biblioteca está construida sobre `pandas` para un rendimiento óptimo a través de operaciones vectorizadas y proporciona una API sencilla para integrar en cualquier proyecto de análisis financiero.

## Descripción General

Esta herramienta toma un DataFrame de `pandas` con datos de velas (OHLCV) y calcula los siguientes indicadores clave del sistema Profitunity de Bill Williams:

-   **Alligator**: Identifica la tendencia y su dirección usando tres medias móviles suavizadas y desplazadas.
-   **Awesome Oscillator (AO)**: Mide el momentum del mercado.
-   **Accelerator Oscillator (AC)**: Mide la aceleración o desaceleración del momentum actual.
-   **Barras Divergentes (Bullish/Bearish)**: Señales de entrada basadas en la divergencia entre el precio y el momentum.
-   **Ventanas de Profitunity (MFI)**: Clasifica las barras en cuatro tipos (Green, Squat, Fade, Fake) para calificar las oportunidades de trading.

## Características

-   **Alto Rendimiento**: Utiliza operaciones vectorizadas de `pandas` y `numpy`, eliminando bucles lentos y siendo eficiente incluso con grandes conjuntos de datos.
-   **API Sencilla**: Instancia la clase, llama a un único método (`calculate_all()`) y obtén un DataFrame con todos los indicadores calculados.
-   **Visualización Integrada**: Incluye métodos para generar gráficos claros y listos para usar del Alligator y los osciladores.
-   **Código Limpio y Mantenible**: Escrito con las mejores prácticas de Python, incluyendo `type hinting` y `docstrings`.
-   **Indicadores Adicionales**: Incluye el Awesome Oscillator (AO) y el Accelerator Oscillator (AC), que son fundamentales en el sistema de Williams.

## Requisitos

-   Python 3.7+
-   pandas
-   numpy
-   matplotlib

## Instalación

1.  Clona este repositorio o descarga el archivo `williams_indicators.py` en tu proyecto.

    ```bash
    git clone https://github.com/luisagreda-aidev/WilliamsIndicators.git
    ```

2.  Asegúrate de tener las dependencias instaladas. Puedes instalarlas usando pip:

    ```bash
    pip install pandas numpy matplotlib
    ```

    O, si tienes un archivo `requirements.txt`:

    ```
    # requirements.txt
    pandas
    numpy
    matplotlib
    ```

    ```bash
    pip install -r requirements.txt
    ```

## Uso Básico

El flujo de trabajo es simple: prepara tus datos, instancia la clase, calcula los indicadores y analiza los resultados.

```python
import pandas as pd
from williams_indicators import WilliamsIndicators # Asumiendo que guardaste la clase en este archivo

# 1. Carga o crea tus datos de velas en un DataFrame de pandas.
# El DataFrame DEBE contener columnas 'Open', 'High', 'Low', 'Close', 'Volume'.
# (Los nombres de las columnas no distinguen entre mayúsculas y minúsculas).
data = pd.read_csv('tus_datos_ohlcv.csv')

# 2. Instancia la clase con tus datos.
indicator_calculator = WilliamsIndicators(data)

# 3. Llama al método `calculate_all()` para obtener un nuevo DataFrame
#    con todos los indicadores añadidos como columnas.
results_df = indicator_calculator.calculate_all()

# 4. ¡Listo! Ahora puedes analizar los resultados.
print("DataFrame con indicadores (últimas 5 filas):")
print(results_df.tail())

# Columnas añadidas:
# - alligator_jaw, alligator_teeth, alligator_lips
# - awesome_oscillator, accelerator_oscillator
# - is_bull_divergent_bar, is_bear_divergent_bar
# - profitunity_window

# 5. Genera gráficos para visualizar los indicadores.
# Grafica los últimos 200 periodos.
indicator_calculator.plot_alligator(num_records=200)
indicator_calculator.plot_oscillators(num_records=200)
```

## API de la Clase

### `WilliamsIndicators(candles_df: pd.DataFrame)`

Constructor de la clase.

-   `candles_df`: Un DataFrame de `pandas` que contiene los datos históricos de las velas. Debe incluir las columnas `Open`, `High`, `Low`, `Close`, y `Volume`.

### `calculate_all() -> pd.DataFrame`

El método principal que calcula todos los indicadores y los añade como nuevas columnas al DataFrame.

-   **Retorna**: Un DataFrame de `pandas` con los indicadores calculados.

### `plot_alligator(num_records: int = 100)`

Genera un gráfico del indicador Alligator superpuesto al precio de cierre.

-   `num_records`: El número de los últimos registros a graficar.

### `plot_oscillators(num_records: int = 100)`

Genera un gráfico con dos subgráficos: el Awesome Oscillator (AO) y el Accelerator Oscillator (AC).

-   `num_records`: El número de los últimos registros a graficar.

## Visualizaciones de Ejemplo

#### Alligator Plot

El gráfico muestra el precio de cierre junto con las tres líneas del Alligator:
-   **Mandíbula (Jaw, azul)**: La línea más lenta, representa el equilibrio a largo plazo.
-   **Dientes (Teeth, rojo)**: La línea intermedia.
-   **Labios (Lips, verde)**: La línea más rápida.

Cuando las líneas están entrelazadas y juntas, el "Alligator duerme" (mercado en rango). Cuando se separan y se mueven en paralelo, el "Alligator despierta" y se alimenta (mercado en tendencia).

  <!-- Reemplaza con una URL de imagen real si la tienes -->

#### Oscillators Plot

Este gráfico muestra el momentum del mercado:
-   **Awesome Oscillator (AO)**: Un histograma que mide el momentum. Verde indica un valor mayor que el anterior, rojo indica un valor menor.
-   **Accelerator Oscillator (AC)**: Mide la aceleración del momentum. Es una señal temprana de posibles cambios en la tendencia.

 <!-- Reemplaza con una URL de imagen real si la tienes -->

## Contribuciones

Las contribuciones son bienvenidas. Si encuentras un error, tienes una sugerencia de mejora o quieres añadir una nueva característica, por favor, abre un "issue" o envía un "pull request".

## Licencia

Este proyecto está bajo la Licencia MIT. Consulta el archivo `LICENSE` para más detalles.
