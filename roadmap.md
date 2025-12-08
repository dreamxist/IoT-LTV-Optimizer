Este es el **Roadmap Técnico Detallado** para *OmniEvo*. Lo he estructurado pensando en un flujo de desarrollo de software ágil pero con el rigor de una investigación académica.

Dividiremos el trabajo en **4 Sprints** (fases) lógicas. Asumiremos que trabajarás en **Python**.

---

## 📅 Sprint 1: El Simulador de "Realidad" (Data Generation)
**Objetivo:** Crear el "Dios" del sistema. Necesitamos generar datos que parezcan reales para tener una "Verdad Terreno" (Ground Truth) contra la cual entrenar tu IA. Sin datos, no hay proyecto.

### Tarea 1.1: Definición del Modelo de Datos (Schema)
No escribas código aún. Define qué estructura tendrán tus datos.
* **Usuarios:** ID, Segmento (VIP, General, Fanático), Presupuesto Total (LTV Real - Variable Oculta).
* **Canales (Inputs):** Define los 5-8 canales que vas a rastrear.
    * *Digital:* `Facebook_Ads`, `Email_Newsletter`, `Google Search`.
    * *App:* `App_Open`, `Ticket_Purchase`.
    * *IoT/Físico:* `RFID_Entrance`, `Main_Stage_Presence` (tiempo en minutos), `Bar_NFC_Payment`.

### Tarea 1.2: El Motor de Simulación Estocástica
Crea un script `data_generator.py`. Este script no debe ser aleatorio puro (`random`), debe ser **probabilístico**.
* **Lógica de Segmentos:**
    * Si el usuario es "VIP", tiene 90% de probabilidad de tener `Ticket_Purchase` alto y `RFID_Entrance`.
    * Si el usuario es "Digital Browser", tiene muchos `Facebook_Ads` pero 0 `RFID_Entrance`.
* **Generación de Ruido:** Introduce aleatoriedad. Un usuario puede ver un anuncio y no comprar. Otro puede comprar sin ver anuncios (tráfico directo).
* **Output:** Generar un DataFrame de Pandas o CSV con 1,000 - 10,000 usuarios simulados.
    * *Columnas:* `user_id`, `ch_fb`, `ch_email`, `ch_rfid`, ..., `LTV_REAL` (Target).

### Tarea 1.3: Normalización de Datos
* Las escalas son diferentes (ej: `Facebook_Ads` son contadores 1-5, `Main_Stage_Presence` son minutos 0-120).
* **Acción:** Aplica *Min-Max Scaling* o *Z-Score* a las columnas de entrada para que el Algoritmo Genético no le de más peso al tiempo solo porque el número es más grande.

---

## 🧬 Sprint 2: Ingeniería del Algoritmo Genético (The Core)
**Objetivo:** Construir el cerebro evolutivo usando una librería como **DEAP** o **PyGAD**.

### Tarea 2.1: Codificación del Cromosoma
* Define la estructura del individuo.
* **Estructura:** Un arreglo de *floats* de tamaño $N$ (donde $N$ es el número de canales).
* **Restricción Crítica:** Implementa una función de reparación o penalización. La suma de los pesos **debe** ser 1 (100% de atribución).
    * *Técnica sugerida:* Softmax o normalización simple ($w_i / \sum w$) después de cada mutación.

### Tarea 2.2: Función de Fitness (La brújula)
Esta es la parte más importante. El algoritmo necesita saber si va bien o mal.
* **Input:** Un individuo (vector de pesos candidatos).
* **Proceso:**
    1.  Toma el dataset de entrenamiento (sin la columna LTV Real).
    2.  Multiplica las interacciones de cada usuario por los pesos del individuo.
    3.  Suma los resultados para obtener el `LTV_Predicho`.
    4.  Compara `LTV_Predicho` vs `LTV_Real` de todo el dataset.
* **Métrica:** Calcula el **RMSE** (Root Mean Square Error).
* **Return:** El valor negativo del RMSE (porque las librerías genéticas suelen buscar *maximizar* el fitness, y queremos *minimizar* el error).

### Tarea 2.3: Configuración de Operadores Evolutivos
Selecciona los métodos matemáticos para la evolución:
* **Selección:** `Tournament Selection` (Torneo). Es robusto y evita convergencia prematura.
* **Cruce (Crossover):** `Simulated Binary Crossover` (SBX) o `Blend Crossover` (BLX-alpha). No uses cruce de un punto simple, porque estamos trabajando con números reales, no binarios.
* **Mutación:** `Gaussian Mutation` (añadir ruido gaussiano con media 0 y desviación pequeña). Esto "ajusta" los pesos finamente.

---

## 🧪 Sprint 3: Experimentación y Ajuste (Hyperparameter Tuning)
**Objetivo:** Hacer que el modelo converja y no se quede estancado.

### Tarea 3.1: Loop de Entrenamiento
* Configura el bucle principal:
    * Población inicial: 50-100 individuos.
    * Generaciones: 50-200.
    * Probabilidad de Cruce ($P_c$): 0.7 - 0.9.
    * Probabilidad de Mutación ($P_m$): 0.05 - 0.2.

### Tarea 3.2: Monitoreo de Convergencia
* Implementa "Logs". En cada generación, guarda:
    * El Fitness del mejor individuo.
    * El Fitness promedio de la población.
* **Gráfica:** Genera un gráfico de línea (Matplotlib). Si la línea se aplana muy rápido, tienes "Convergencia Prematura" (aumenta mutación). Si oscila demasiado y no baja, reduce la mutación.

### Tarea 3.3: Implementación de "Elitismo"
* Asegúrate de que el mejor individuo de la Generación X pase inalterado a la Generación X+1. Esto garantiza que nunca pierdas tu mejor solución por culpa de una mala mutación.

---

## 📊 Sprint 4: Validación y Benchmarking (La Tesis)
**Objetivo:** Demostrar que tu IA sirve para algo comparándola con métodos "tontos".

### Tarea 4.1: Implementar Modelos Base (Dummy Models)
Crea dos funciones simples para comparar:
1.  **Modelo Lineal:** Asigna peso $1/N$ a todos los canales. Calcula el RMSE.
2.  **Modelo Last-Touch:** Asigna peso 1.0 al último canal con interacción y 0 al resto. Calcula el RMSE.

### Tarea 4.2: Comparación Estadística
* Ejecuta tu mejor individuo (el ganador del Genético) sobre el **Set de Test** (datos que el algoritmo nunca vio durante el entrenamiento).
* Compara el RMSE del Genético vs. el RMSE del Lineal y Last-Touch.
* **Éxito:** Tu RMSE debe ser menor. Calcula el % de mejora:
    $$Mejora = \frac{RMSE_{Lineal} - RMSE_{Genetico}}{RMSE_{Lineal}} \times 100\%$$

### Tarea 4.3: Interpretación de Negocio (El "Storytelling")
* Extrae el vector de pesos ganador. Ejemplo: `[Facebook: 0.1, App: 0.3, RFID: 0.6]`.
* **Análisis:** "El algoritmo descubrió que la interacción física (RFID) predice el LTV 2 veces mejor que la App y 6 veces mejor que Facebook Ads".
* Esta es la conclusión que valida tu proyecto de Soft Computing.

---

### 🛠 Herramientas Sugeridas

* **Lenguaje:** Python 3.x
* **Librería GA:** `DEAP` (Es la estándar académica, muy flexible) o `PyGAD` (Más moderna y fácil de usar).
* **Datos:** `Pandas`, `Numpy`.
* **Gráficos:** `Matplotlib` o `Seaborn`.

