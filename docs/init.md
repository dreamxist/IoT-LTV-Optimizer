# Optimización Metaheurística de Atribución Omnicanal

> **Proyecto:** OmniEvo - Optimización de Atribución Omnicanal con Algoritmos Genéticos
>
> **Paper Base:** Pramono et al. (2019) - "Estimating Customer Segmentation based on Customer Lifetime Value Using Two-Stage Clustering Method" ([IEEE](https://ieeexplore.ieee.org/document/8887704/))
>
> **Documentación relacionada:** [Análisis del Paper Base](paper_base.md)

---

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

### 1.3 Relación con el Paper Base

El paper de Pramono et al. (2019) propone usar **Fuzzy AHP** para determinar pesos de variables LRFM en segmentación de clientes. **OmniEvo extiende y mejora este enfoque:**

| Aspecto | Paper Original | OmniEvo |
|---------|----------------|---------|
| **Variables** | LRFM (4 variables) | Canales omnicanal (9+ variables) |
| **Determinación de pesos** | Fuzzy AHP (expertos) | **Algoritmo Genético** (data-driven) |
| **Objetivo** | Segmentación (clustering) | Predicción de LTV |
| **Fuente de datos** | Transacciones | Digital + App + **IoT** |
| **Adaptabilidad** | Estático | Dinámico (re-optimizable) |

**Innovación clave:** Eliminamos la subjetividad de Fuzzy AHP y permitimos que los datos determinen los pesos óptimos mediante evolución artificial.

---

## 2. Definición del Problema de Investigación

### 2.1 Enunciado Formal

El problema consiste en la **optimización de un vector de pesos de atribución $\vec{w}$** en un espacio de búsqueda continuo $n$-dimensional, donde $n$ es el número de canales o *touchpoints* rastreados.

El objetivo es minimizar la función de costo $J(\vec{w})$, definida como el error entre el *Customer Lifetime Value* (LTV) modelado y el LTV observado históricamente. Dado que la relación entre interacciones físicas (IoT) y comportamientos digitales es altamente no lineal y ruidosa, los métodos de optimización clásica (como el descenso de gradiente) pueden quedar atrapados en mínimos locales o requerir funciones diferenciables que no poseemos.

### 2.2 Pregunta de Investigación

> ¿Puede un Algoritmo Genético (AG) evolucionar una estrategia de atribución (pesos) que supere estadísticamente la precisión predictiva de los modelos heurísticos tradicionales en un entorno Omnicanal simulado?

### 2.3 Hipótesis

**H₀ (Nula):** Los pesos evolucionados por AG no difieren significativamente de una distribución uniforme (1/N) en capacidad predictiva.

**H₁ (Alternativa):** Los pesos evolucionados por AG producen un RMSE significativamente menor que los modelos baseline, demostrando que la atribución data-driven supera a los métodos heurísticos.

---

## 3. Escenario de Simulación: "Smart Music Festival"

Para evitar las limitaciones de acceso a datos privados, se desarrollará un generador de datos sintéticos que simule el ecosistema de un festival de música inteligente.

### 3.1 Arquitectura de Datos Simulada

El sistema genera datos representando tres capas de interacción:

```
┌─────────────────────────────────────────────────────────────────┐
│                    SMART MUSIC FESTIVAL                         │
├─────────────────────────────────────────────────────────────────┤
│  CAPA DIGITAL (AdTech)                                          │
│  ├── facebook_ads      → Impresiones de anuncios               │
│  ├── email_opens       → Apertura de newsletters                │
│  └── web_visits        → Navegación en sitio del festival       │
├─────────────────────────────────────────────────────────────────┤
│  CAPA TRANSACCIONAL (App/Backend)                               │
│  ├── app_opens         → Uso de la aplicación oficial          │
│  ├── ticket_purchase   → Compra de entrada (Conversión Macro)  │
│  └── wallet_topup      → Carga de pulsera cashless              │
├─────────────────────────────────────────────────────────────────┤
│  CAPA FÍSICA (IoT & Edge Computing)                             │
│  ├── rfid_entry        → Validación en pórticos RFID           │
│  ├── stage_dwell_time  → Tiempo en escenarios (BLE/Wi-Fi)      │
│  └── nfc_purchases     → Compras con pulsera NFC               │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Segmentos de Usuario

El generador produce usuarios con comportamientos diferenciados:

| Segmento | Prob. | LTV Range | Comportamiento |
|----------|-------|-----------|----------------|
| **VIP** | 15% | $80-150 | Alta actividad en todos los canales |
| **Regular** | 55% | $30-80 | Mix balanceado de interacciones |
| **Digital-Only** | 20% | $5-30 | Solo canales digitales, sin IoT |
| **Casual** | 10% | $0-20 | Baja actividad general |

### 3.3 El Desafío del "Server-Side"

Se asume que todos los eventos son enviados a un contenedor *Server-Side* que unifica la identidad del usuario (User Stitching). El dataset final es una matriz donde:
- Cada **fila** es un usuario
- Cada **columna** es la frecuencia/intensidad de interacción en un canal
- La última columna es el **LTV real** (variable objetivo)

---

## 4. Diseño de la Solución (Soft Computing)

El núcleo del proyecto es un **Algoritmo Genético (AG)** diseñado para explorar el espacio de soluciones de atribución.

### 4.1 Representación del Individuo (Genotipo)

Cada solución candidata (individuo) es un vector de números reales:

$$I = [w_1, w_2, ..., w_n]$$

**Ejemplo con 9 canales:**
```python
individuo = [0.05, 0.12, 0.03, 0.08, 0.10, 0.15, 0.07, 0.10, 0.30]
#            fb    email web   app   ticket wallet rfid  stage nfc
```

**Restricción de Normalización:**
$$\sum_{i=1}^{n} w_i = 1 \quad \text{y} \quad w_i \geq 0 \quad \forall i$$

Esta restricción asegura que los pesos sean interpretables como una distribución de crédito.

### 4.2 Función de Aptitud (Fitness Function)

Para un individuo $\vec{w}$ y un conjunto de usuarios $U$:

1. **LTV Predicho** para cada usuario $u$:
   $$LTV_{pred}^{(u)} = \left( \sum_{i=1}^{n} w_i \cdot x_i^{(u)} \right) \times \text{scale}$$

2. **Error (RMSE):**
   $$\text{RMSE} = \sqrt{\frac{1}{|U|} \sum_{u \in U} (LTV_{pred}^{(u)} - LTV_{real}^{(u)})^2}$$

3. **Fitness** (DEAP maximiza, por eso negamos RMSE):
   $$\text{Fitness}(\vec{w}) = -\text{RMSE}$$

### 4.3 Operadores Evolutivos

| Operador | Método | Parámetros |
|----------|--------|------------|
| **Selección** | Tournament Selection | k=3 |
| **Cruce** | Blend Crossover (BLX-α) | α=0.5 |
| **Mutación** | Gaussian Mutation | σ=0.1, indpb=0.3 |
| **Elitismo** | Preservar mejores | n=2 |
| **Reparación** | Normalización post-operador | Σwᵢ = 1 |

### 4.4 Comparación con Fuzzy AHP (Paper Base)

```
Paper Original (Fuzzy AHP):
  Expertos → Comparación pareada → Matriz → Pesos fijos
  [Subjetivo, estático, limitado a ~5 variables]

OmniEvo (Algoritmo Genético):
  Datos → Población inicial → Evolución → Pesos óptimos
  [Objetivo, dinámico, escalable a N variables]
```

---

## 5. Metodología de Validación

### 5.1 Split de Datos
- **70% Entrenamiento:** Para ejecutar el AG
- **30% Test:** Para validar los pesos finales (datos nunca vistos)

### 5.2 Modelos Baseline

| Modelo | Descripción | Propósito |
|--------|-------------|-----------|
| **Uniforme** | wᵢ = 1/n | Hipótesis nula |
| **Last-Touch** | w_último = 1, resto = 0 | Modelo tradicional |
| **First-Touch** | w_primero = 1, resto = 0 | Modelo alternativo |
| **Random** | wᵢ ~ U(0,1) normalizado | Control negativo |
| **Position Decay** | wᵢ = decay^i | Modelo decreciente |

### 5.3 Métricas de Éxito

| Métrica | Fórmula | Objetivo |
|---------|---------|----------|
| **RMSE** | √(Σ(y-ŷ)²/n) | Minimizar |
| **Pearson r** | corr(y, ŷ) | Maximizar → 1 |
| **Mejora %** | (RMSE_base - RMSE_GA) / RMSE_base × 100 | Maximizar |

---

## 6. Stack Tecnológico

| Componente | Tecnología | Propósito |
|------------|------------|-----------|
| **Lenguaje** | Python 3.9+ | Base |
| **AG** | DEAP | Algoritmos evolutivos |
| **Datos** | Pandas, NumPy | Manipulación y álgebra |
| **ML** | scikit-learn | Métricas y normalización |
| **Visualización** | Matplotlib, Seaborn | Gráficos |
| **Testing** | pytest | Validación de código |

---

## 7. Roadmap de Desarrollo

| Sprint | Objetivo | Estado |
|--------|----------|--------|
| **Sprint 1** | Generador de datos sintéticos | ✅ Completado |
| **Sprint 2** | Core del Algoritmo Genético | ✅ Completado |
| **Sprint 3** | Experimentación y tuning | 🔄 En progreso |
| **Sprint 4** | Validación y benchmarking | ⏳ Pendiente |

Ver [roadmap.md](roadmap.md) para detalles de cada sprint.

---

## 8. Referencias

1. **Paper Base:** Pramono, P. P., Surjandari, I., & Laoh, E. (2019). *Estimating Customer Segmentation based on Customer Lifetime Value Using Two-Stage Clustering Method*. IEEE ICSSSM 2019. [DOI](https://ieeexplore.ieee.org/document/8887704/)

2. **LRFM Model:** Alvandi, M., Fazli, S., & Abdoli, F. (2012). *K-Mean Clustering Method For Analysis Customer Lifetime Value With LRFM Relationship Model*.

3. **Algoritmos Genéticos:** Eiben, A. E., & Smith, J. E. (2015). *Introduction to Evolutionary Computing*. Springer.

4. **DEAP:** Fortin, F. A., et al. (2012). *DEAP: Evolutionary Algorithms Made Easy*. Journal of Machine Learning Research.
