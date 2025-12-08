
# Optimización Metaheurística de Atribución Omnicanal


## 1. Introducción y Contextualización

### 1.1 El Panorama Actual
La industria del marketing digital enfrenta una crisis de datos. Con la eliminación de las *cookies* de terceros y las restricciones de privacidad (iOS 14+), las empresas han perdido visibilidad sobre el comportamiento del usuario. Sin embargo, el desafío mayor no es solo digital: es **híbrido**.

En eventos masivos, retail y ciudades inteligentes, el usuario transita constantemente entre el mundo físico y el digital. Un asistente a un festival puede ver un anuncio en Instagram, descargar una App, pero realizar la conversión final (gasto) mediante una pulsera NFC en un punto de venta físico.

### 1.2 La Brecha Técnica
Actualmente, los sistemas de análisis operan en silos. Google Analytics mide la web; los sistemas POS miden la caja. Unir estos puntos requiere **Ingeniería de Atribución**. Sin embargo, incluso con los datos unidos, persiste una pregunta matemática sin resolver: **¿Qué evento causó realmente la compra?**

Los modelos actuales son deterministas y arbitrarios:
* *Last Click:* Le da todo el crédito al último evento (ignorando la construcción de marca).
* *Lineal:* Reparte el crédito equitativamente (ignorando que ciertos eventos son más influyentes que otros).

Este proyecto propone una solución basada en **Soft Computing** para encontrar los pesos de atribución "verdaderos" que maximicen la predicción del valor del cliente.

---

## 2. Definición del Problema de Investigación

### 2.1 Enunciado Formal
El problema consiste en la **optimización de un vector de pesos de atribución $\vec{w}$** en un espacio de búsqueda continuo $n$-dimensional, donde $n$ es el número de canales o *touchpoints* rastreados.

El objetivo es minimizar la función de costo $J(\vec{w})$, definida como el error entre el *Customer Lifetime Value* (LTV) modelado y el LTV observado históricamente. Dado que la relación entre interacciones físicas (IoT) y comportamientos digitales es altamente no lineal y ruidosa, los métodos de optimización clásica (como el descenso de gradiente) pueden quedar atrapados en mínimos locales o requerir funciones diferenciables que no poseemos.

### 2.2 Pregunta de Investigación
> ¿Puede un Algoritmo Genético (AG) evolucionar una estrategia de atribución (pesos) que supere estadísticamente la precisión predictiva de los modelos heurísticos tradicionales en un entorno Omnicanal simulado?

---

## 3. Escenario de Simulación: "Smart Music Festival"

Para evitar las limitaciones de acceso a datos privados, se desarrollará un generador de datos sintéticos que simule el ecosistema de un festival de música inteligente.

### 3.1 Arquitectura de Datos Simulada
El sistema generará `JSON` logs representando tres capas de interacción:

1.  **Capa Digital (AdTech):**
    * `ad_impression`: El usuario ve un anuncio.
    * `social_share`: El usuario comparte el evento.
    * `web_visit`: Navegación en el sitio del festival.
2.  **Capa Transaccional (App/Backend):**
    * `app_install`: Instalación de la aplicación oficial.
    * `ticket_purchase`: Compra de la entrada (Conversión Macro).
    * `wallet_topup`: Carga de dinero en la pulsera *cashless*.
3.  **Capa Física (IoT & Edge Computing):**
    * `rfid_entry`: Validación de entrada en pórticos.
    * `stage_dwell_time`: Tiempo de permanencia en un escenario (capturado por balizas BLE/Wi-Fi).
    * `nfc_purchase`: Compra de comida/bebida/merch (Conversión Micro).

### 3.2 El Desafío del "Server-Side"
Se asumirá que todos estos eventos son enviados a un contenedor *Server-Side* que unifica la identidad del usuario (User Stitching). El dataset final será una matriz donde cada fila es un usuario y cada columna es la frecuencia o intensidad de sus interacciones en cada canal, más su LTV final (variable objetivo).

---

## 4. Diseño de la Solución (Soft Computing)

El núcleo del proyecto es un **Algoritmo Genético (AG)** diseñado para explorar el espacio de soluciones de atribución.

### 4.1 Representación del Individuo (Genotipo)
Cada solución candidata (individuo) será un vector de números reales flotantes.
$$I = [w_{ad}, w_{social}, w_{web}, w_{app}, w_{entry}, w_{dwell}, w_{nfc}]$$

**Restricciones:**
Para que el modelo sea interpretable como una "distribución de crédito", se debe cumplir la restricción de normalización en cada evaluación:
$$\sum_{i=1}^{n} w_i = 1 \quad \text{y} \quad 0 \leq w_i \leq 1$$

### 4.2 Función de Aptitud (Fitness Function)
La función de fitness evaluará la capacidad predictiva del vector de pesos.
Para un individuo $\vec{w}$ y un conjunto de usuarios $U$:

1.  Se calcula el **LTV Predicho** para cada usuario $u$ como el producto punto entre sus interacciones ($\vec{x}_u$) y los pesos ($\vec{w}$):
    $$LTV_{pred}^{(u)} = \vec{w} \cdot \vec{x}_u \times \text{FactorDeEscala}$$
2.  Se compara contra el **LTV Real** ($y_u$).
3.  Se calcula el **RMSE (Root Mean Square Error)**:
    $$\text{Fitness}(\vec{w}) = \frac{1}{\text{RMSE}} = \left( \sqrt{\frac{1}{|U|} \sum_{u \in U} (LTV_{pred}^{(u)} - y_u)^2} \right)^{-1}$$
    *(Nota: Maximizamos el inverso del error, o minimizamos el error directamente).*

### 4.3 Operadores Evolutivos
* **Selección:** *Tournament Selection* (Tamaño k=3). Permite presión selectiva manteniendo diversidad.
* **Cruce (Crossover):** *Simulated Binary Crossover (SBX)* o *Blend Crossover (BLX-$\alpha$)*. Estos son operadores específicos para codificación real que permiten generar descendencia cerca de los padres pero explorando el espacio intermedio.
* **Mutación:** *Gaussian Mutation*. Se añade un pequeño valor aleatorio derivado de una distribución normal a un gen seleccionado al azar, permitiendo ajustes finos.
* **Reparación:** Post-operadores, se debe re-normalizar el vector para que sume 1 nuevamente.

---

## 5. Metodología de Validación

Para demostrar la validez científica del proyecto, se utilizará un esquema de validación cruzada:

1.  **Split de Datos:** 70% Entrenamiento (para correr el AG), 30% Prueba (para validar los pesos finales).
2.  **Modelos Base (Baselines):** Se comparará el desempeño del AG contra:
    * **Modelo Aleatorio:** Pesos random (Control negativo).
    * **Modelo Uniforme:** Todos los pesos son iguales ($1/n$).
    * **Modelo Last-Interaction:** Peso 1.0 al último evento, 0.0 al resto.
3.  **Métricas de Éxito:**
    * Reducción porcentual del RMSE respecto a los modelos base.
    * Coeficiente de Correlación de Pearson ($r$) en el set de prueba.

---

## 6. Stack Tecnológico Propuesto

* **Lenguaje:** Python 3.9+
* **Librerías de Soft Computing:** `DEAP` (Distributed Evolutionary Algorithms in Python) o `PyGAD`.
* **Procesamiento de Datos:** `Pandas` (Dataframes), `NumPy` (Álgebra lineal).
* **Visualización:** `Matplotlib` / `Seaborn` (Para graficar la convergencia del fitness a través de las generaciones).

---

## 7. Plan de Trabajo (Roadmap)

* **Semana 1:** Implementación del generador de datos sintéticos ("Smart Music Festival Simulator"). Definición de distribuciones de probabilidad para los usuarios.
* **Semana 2:** Desarrollo del motor del Algoritmo Genético (Configuración de DEAP, definición de cromosomas y fitness).
* **Semana 3:** Experimentación y ajuste de hiperparámetros (Tamaño de población, probabilidad de mutación, número de generaciones).
* **Semana 4:** Análisis de resultados, comparación con baselines, generación de gráficos de convergencia y redacción del informe final.
