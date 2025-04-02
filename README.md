# Backtesting de Estrategia de Trading para Ethereum con Machine Learning

## Objetivo General

Desarrollar, entrenar y evaluar automáticamente una estrategia de trading algorítmica para Ethereum (ETH-USD) utilizando datos históricos de mercado y un modelo de Machine Learning (Random Forest).

## Proceso Implementado

1.  **Adquisición de Datos:** Descarga de datos históricos diarios (OHLCV - Apertura, Máximo, Mínimo, Cierre, Volumen) para ETH-USD de los últimos 5 años utilizando la librería `yfinance`.
2.  **Ingeniería de Características (Indicadores Técnicos):** Cálculo de un conjunto de indicadores técnicos comunes usando `Pandas` sobre los datos históricos:
    *   Medias Móviles Exponenciales (EMAs: 10, 50, 100 días)
    *   Índice de Fuerza Relativa (RSI)
    *   Convergencia/Divergencia de la Media Móvil (MACD)
    *   Bandas de Bollinger
    *   Rango Verdadero Promedio (ATR)
    *   Volumen On-Balance (OBV)
    *   Índice Direccional Promedio (ADX)
    *   Retorno del día anterior.
3.  **Entrenamiento del Modelo Predictivo:**
    *   **Modelo:** Se utiliza un `RandomForestClassifier` de `Scikit-learn`.
    *   **Objetivo:** Predecir la dirección del precio del día siguiente (clasificación binaria: Sube/Baja) basándose en los indicadores técnicos del día actual.
    *   **Optimización y Validación:** Se realiza una búsqueda de hiperparámetros (`RandomizedSearchCV`) con validación cruzada específica para series temporales (`TimeSeriesSplit`) para encontrar la mejor configuración del modelo y evitar el sobreajuste a datos futuros.
4.  **Evaluación del Modelo:** Medición de la capacidad predictiva del modelo entrenado en un conjunto de datos de prueba (hold-out set) utilizando métricas como el **AUC (Area Under the Curve)**.
5.  **Simulación de Estrategia (Backtesting):**
    *   **Lógica:** Se simulan operaciones de compra/venta basadas en las probabilidades predichas por el modelo.
    *   **Filtro de Tendencia:** Las operaciones solo se consideran si están alineadas con la tendencia general del mercado (precio > EMA 100 para compras, precio < EMA 100 para ventas/cortos).
    *   **Umbrales de Señal:** Se definen umbrales de probabilidad (ej. > 60% para comprar, < 40% para vender) para generar las señales de trading.
    *   **Costos de Transacción:** Se incluye una estimación de costos (comisiones, slippage) por cada operación simulada.
6.  **Análisis de Rendimiento:**
    *   **Visualización:** Gráficos de la evolución del capital de la estrategia vs. un benchmark de "Comprar y Mantener" (Buy & Hold).
    *   **Métricas Clave:** Cálculo del Retorno Neto Total, Sharpe Ratio (retorno ajustado al riesgo) y Máximo Drawdown (peor caída porcentual).

## Tecnologías Utilizadas

*   **Lenguaje:** Python 3
*   **Manipulación de Datos:** `Pandas`, `NumPy`
*   **Adquisición de Datos:** `yfinance`
*   **Machine Learning:** `Scikit-learn` (RandomForestClassifier, RandomizedSearchCV, TimeSeriesSplit, metrics)
*   **Visualización:** `Matplotlib`, `Seaborn` (o similar)
*   **Gestión de Dependencias:** `venv`, `pip`, `requirements.txt` (*Asegúrate de crear este archivo con `pip freeze > requirements.txt`*)

## Instalación y Uso (Ejemplo)

1.  **Clonar el repositorio:**
    ```bash
    git clone https://github.com/TamaraKaren/ML_Finanzas.git
    cd ML_Finanzas
    ```
2.  **Crear y activar un entorno virtual:**
    ```bash
    python -m venv venv
    # En Windows:
    .\venv\Scripts\activate
    # En macOS/Linux:
    source venv/bin/activate
    ```
3.  **Instalar dependencias:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Ejecutar el script/notebook principal:**
    ```bash
    # Si es un script .py
    python nombre_del_script.py
    # Si es un notebook .ipynb
    jupyter notebook ML_Finanzas.ipynb
    ```

## Habilidades Demostradas y Relevancia

Este proyecto demuestra experiencia en:

*   El **ciclo de vida completo de un proyecto de datos y Machine Learning**: desde la adquisición y limpieza, pasando por la ingeniería de características, hasta el entrenamiento, validación y aplicación del modelo.
*   **Manejo avanzado de datos tabulares y series temporales** con `Pandas`.
*   Aplicación de técnicas de **Análisis Técnico** financiero.
*   Implementación y **optimización de modelos de Machine Learning (`Scikit-learn`)** con conciencia de las particularidades de los datos temporales (`TimeSeriesSplit`).
*   **Evaluación rigurosa** del rendimiento de modelos y estrategias.
*   **Traducción de insights de modelos a acciones simuladas** (backtesting).

**Este flujo de trabajo es directamente análogo y fundamental para preparar datos empresariales (a menudo desordenados) y utilizarlos eficazmente en sistemas de IA más avanzados, incluyendo la IA Generativa.** La calidad de los datos de entrada y la relevancia de las características (como los indicadores aquí) son cruciales para la fiabilidad de los modelos GenAI, ya sea para construir bases de conocimiento precisas para **sistemas RAG (Retrieval-Augmented Generation)** o para realizar un **fine-tuning** efectivo de modelos pre-entrenados.


# Backtesting an Ethereum Trading Strategy with Machine Learning

## General Objective

To automatically develop, train, and evaluate an algorithmic trading strategy for Ethereum (ETH-USD) using historical market data and a Machine Learning model (Random Forest).

## Implemented Process

1.  **Data Acquisition:** Download historical daily data (OHLCV - Open, High, Low, Close, Volume) for ETH-USD over the last 5 years using the `yfinance` library.
2.  **Feature Engineering (Technical Indicators):** Calculate a set of common technical indicators using `Pandas` on the historical data:
    *   Exponential Moving Averages (EMAs: 10, 50, 100 days)
    *   Relative Strength Index (RSI)
    *   Moving Average Convergence Divergence (MACD)
    *   Bollinger Bands
    *   Average True Range (ATR)
    *   On-Balance Volume (OBV)
    *   Average Directional Index (ADX)
    *   Previous day's return.
3.  **Predictive Model Training:**
    *   **Model:** A `RandomForestClassifier` from `Scikit-learn` is used.
    *   **Objective:** Predict the next day's price direction (binary classification: Up/Down) based on the current day's technical indicators.
    *   **Optimization and Validation:** Hyperparameter tuning (`RandomizedSearchCV`) is performed with time-series specific cross-validation (`TimeSeriesSplit`) to find the best model configuration and prevent overfitting on future data.
4.  **Model Evaluation:** Measure the predictive power of the trained model on a hold-out test set using metrics like **AUC (Area Under the Curve)**.
5.  **Strategy Simulation (Backtesting):**
    *   **Logic:** Simulate buy/sell trades based on the probabilities predicted by the model.
    *   **Trend Filter:** Trades are only considered if aligned with the general market trend (price > 100-day EMA for buys, price < 100-day EMA for sells/shorts).
    *   **Signal Thresholds:** Probability thresholds (e.g., > 60% to buy, < 40% to sell) are defined to generate trading signals.
    *   **Transaction Costs:** An estimate of costs (commissions, slippage) is included for each simulated trade.
6.  **Performance Analysis:**
    *   **Visualization:** Plot the strategy's equity curve compared to a "Buy & Hold" benchmark.
    *   **Key Metrics:** Calculate the Total Net Return, Sharpe Ratio (risk-adjusted return), and Maximum Drawdown (worst peak-to-trough decline).

## Technologies Used

*   **Language:** Python 3
*   **Data Manipulation:** `Pandas`, `NumPy`
*   **Data Acquisition:** `yfinance`
*   **Machine Learning:** `Scikit-learn` (RandomForestClassifier, RandomizedSearchCV, TimeSeriesSplit, metrics)
*   **Visualization:** `Matplotlib`, `Seaborn` (or similar)
*   **Dependency Management:** `venv`, `pip`, `requirements.txt` (*Ensure you create this file using `pip freeze > requirements.txt`*)

## Installation and Usage (Example)

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/TamaraKaren/ML_Finanzas.git
    cd ML_Finanzas
    ```
2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    # On Windows:
    .\venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Run the main script/notebook:**
    ```bash
    # If it's a .py script
    python your_script_name.py
    # If it's an .ipynb notebook
    jupyter notebook ML_Finanzas.ipynb
    ```

## Demonstrated Skills and Relevance

This project demonstrates experience in:

*   The **entire lifecycle of a data and Machine Learning project**: from acquisition and cleaning, through feature engineering, to model training, validation, and application.
*   **Advanced handling of tabular data and time series** with `Pandas`.
*   Application of financial **Technical Analysis** techniques.
*   Implementation and **optimization of Machine Learning models (`Scikit-learn`)** with awareness of time-series data specifics (`TimeSeriesSplit`).
*   **Rigorous evaluation** of model and strategy performance.
*   **Translating model insights into simulated actions** (backtesting).

**This workflow is directly analogous and fundamental to preparing often messy enterprise data for effective use in more advanced AI systems, including Generative AI.** The quality of input data and the relevance of features (like the indicators here) are crucial for the reliability of GenAI models, whether for building accurate knowledge bases for **RAG (Retrieval-Augmented Generation) systems** or for effective **fine-tuning** of pre-trained models.
