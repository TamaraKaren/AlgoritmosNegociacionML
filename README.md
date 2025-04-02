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
    #  .ipynb
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
