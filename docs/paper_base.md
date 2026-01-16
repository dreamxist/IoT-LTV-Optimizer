# Paper Base: Análisis y Relación con OmniEvo

## Referencia Bibliográfica

**Título:** "Estimating Customer Segmentation based on Customer Lifetime Value Using Two-Stage Clustering Method"

**Autores:** Pramono, P. P., Surjandari, I., & Laoh, E.

**Publicación:** 2019 16th International Conference on Service Systems and Service Management (ICSSSM)

**DOI:** [10.1109/ICSSSM.2019.8887704](https://ieeexplore.ieee.org/document/8887704/)

---

## 1. Resumen del Paper

### 1.1 Contexto y Problema

El paper aborda el desafío de la **segmentación de clientes** en la industria de belleza en Indonesia. En un entorno altamente competitivo, las empresas necesitan evaluar y gestionar las interacciones con clientes mediante Customer Relationship Management (CRM) efectivo.

El objetivo principal es **especificar segmentos de clientes con valores de vida similares** para que las empresas puedan aplicar estrategias diferenciadas y apropiadas a cada segmento.

### 1.2 Metodología Propuesta

#### Two-Stage Clustering (Clustering en Dos Etapas)

```
Etapa 1: Ward's Method    →    Determinar número óptimo de clusters (k)
Etapa 2: K-Means          →    Realizar segmentación final con k clusters
```

#### Modelo LRFM

El paper utiliza el modelo **LRFM** (extensión del tradicional RFM) con 4 variables:

| Variable | Nombre | Descripción | Interpretación |
|----------|--------|-------------|----------------|
| **L** | Length | Duración de la relación cliente-empresa | Mayor L → Cliente más leal |
| **R** | Recency | Tiempo desde la última transacción | Menor R → Cliente más activo |
| **F** | Frequency | Número de transacciones en el período | Mayor F → Cliente más frecuente |
| **M** | Monetary | Valor monetario total de transacciones | Mayor M → Cliente más valioso |

#### Extensión LRFM-AI

El paper también evalúa un modelo extendido **LRFM-AI** que incluye:
- **AI** (Average Item): Promedio de ítems por transacción

**Resultado:** La adición de AI no mostró mejora significativa sobre LRFM básico.

### 1.3 Cálculo del CLV

#### Fórmula de Customer Lifetime Value

$$CLV = \sum_{i \in \{L,R,F,M\}} W_i \times N_i$$

Donde:
- $N_i$ = Valor **normalizado** de cada variable LRFM
- $W_i$ = **Peso** de cada variable (importancia relativa)

#### Normalización

Cada variable se normaliza al rango [0, 1]:

$$N_i = \frac{X_i - X_{min}}{X_{max} - X_{min}}$$

#### Determinación de Pesos: Fuzzy AHP

Los pesos $W_i$ se determinan mediante **Fuzzy Analytic Hierarchy Process (Fuzzy AHP)**:

1. Expertos comparan pares de variables (¿L es más importante que R?)
2. Se construye matriz de comparación pareada
3. Se aplica lógica difusa para manejar incertidumbre
4. Se calculan pesos normalizados ($\sum W_i = 1$)

**Ejemplo de pesos típicos obtenidos:**

| Variable | Peso (ejemplo) |
|----------|----------------|
| Frequency | 0.47 |
| Monetary | 0.35 |
| Recency | 0.17 |
| Length | 0.01 |

### 1.4 Resultados Clave

1. El método two-stage clustering produce segmentos más homogéneos
2. LRFM-AI no mejora significativamente sobre LRFM
3. La ponderación por Fuzzy AHP permite priorizar variables según el contexto de negocio
4. Se identifican características de clientes de alto y bajo potencial

---

## 2. Conexión con OmniEvo

### 2.1 Similitudes Fundamentales

| Aspecto | Paper Original | OmniEvo |
|---------|----------------|---------|
| **Objetivo** | Estimar CLV para segmentación | Estimar LTV para atribución |
| **Variables de entrada** | LRFM (4 variables) | Canales omnicanal (9+ variables) |
| **Restricción de pesos** | $\sum W_i = 1$ | $\sum w_i = 1$ |
| **Normalización** | Min-Max Scaling | Min-Max Scaling |
| **Métrica de éxito** | Validez de clusters | RMSE / Correlación Pearson |

### 2.2 Innovación de OmniEvo

#### Problema con Fuzzy AHP

El enfoque del paper original tiene limitaciones:

1. **Subjetividad:** Los pesos dependen de juicio experto
2. **Estático:** Los pesos no se adaptan a nuevos datos
3. **Limitado:** Difícil escalar a muchas variables
4. **Sesgo:** Expertos pueden tener prejuicios sobre canales

#### Solución: Algoritmo Genético

OmniEvo reemplaza Fuzzy AHP con un **Algoritmo Genético**:

```
Paper Original:     Expertos → Fuzzy AHP → Pesos fijos
OmniEvo:           Datos → Algoritmo Genético → Pesos optimizados
```

| Aspecto | Fuzzy AHP | Algoritmo Genético |
|---------|-----------|-------------------|
| **Fuente de pesos** | Juicio experto | Datos históricos |
| **Adaptabilidad** | Estático | Dinámico |
| **Escalabilidad** | ~5 variables | N variables |
| **Objetividad** | Subjetivo | Data-driven |
| **Automatización** | Manual | Automático |

### 2.3 Extensión del Modelo

#### De LRFM a Omnicanal

El paper usa 4 variables transaccionales. OmniEvo extiende a **3 capas de datos**:

```
LRFM (Paper)              →    OmniEvo (Extendido)
─────────────────────────────────────────────────────
Length                    →    Capa Digital
Recency                        - facebook_ads
Frequency                      - email_opens
Monetary                       - web_visits

                          →    Capa App/Transaccional
                               - app_opens
                               - ticket_purchase
                               - wallet_topup

                          →    Capa Física (IoT)
                               - rfid_entry
                               - stage_dwell_time
                               - nfc_purchases
```

#### De Segmentación a Predicción

```
Paper:    LRFM → Clustering → Segmentos → Estrategias por segmento
OmniEvo:  Canales → AG → Pesos óptimos → Predicción de LTV individual
```

---

## 3. Formulación Matemática Comparada

### 3.1 Paper Original (Fuzzy AHP)

$$CLV_u = W_L \cdot N_L^{(u)} + W_R \cdot N_R^{(u)} + W_F \cdot N_F^{(u)} + W_M \cdot N_M^{(u)}$$

Donde los pesos $W$ son determinados por expertos vía Fuzzy AHP.

### 3.2 OmniEvo (Algoritmo Genético)

$$LTV_{pred}^{(u)} = \sum_{i=1}^{n} w_i \cdot x_i^{(u)} \cdot \text{scale}$$

Donde los pesos $w$ son **evolucionados** minimizando:

$$\text{Fitness}(\vec{w}) = -\text{RMSE} = -\sqrt{\frac{1}{|U|} \sum_{u \in U} (LTV_{pred}^{(u)} - LTV_{real}^{(u)})^2}$$

### 3.3 Restricción Común

Ambos enfoques comparten la restricción de normalización:

$$\sum_{i} w_i = 1 \quad \text{y} \quad w_i \geq 0 \quad \forall i$$

En OmniEvo, esto se implementa mediante:

```python
def normalize_weights(weights):
    weights = np.maximum(weights, 0)  # No negativos
    return weights / weights.sum()    # Suma = 1
```

---

## 4. Contribución Científica de OmniEvo

### 4.1 Aportaciones Principales

1. **Data-Driven Attribution:** Elimina la subjetividad de Fuzzy AHP
2. **Omnichannel Integration:** Unifica datos digitales, transaccionales e IoT
3. **Scalability:** Maneja N canales sin intervención experta
4. **Adaptability:** Los pesos pueden re-optimizarse con nuevos datos

### 4.2 Hipótesis de Investigación

> **H₀:** Los pesos evolucionados por AG no difieren significativamente de una distribución uniforme (1/N)
>
> **H₁:** Los pesos evolucionados por AG producen un RMSE significativamente menor que los modelos baseline

### 4.3 Métricas de Validación

| Métrica | Descripción | Objetivo |
|---------|-------------|----------|
| RMSE | Error cuadrático medio | Minimizar |
| Pearson r | Correlación predicho vs real | Maximizar (→1) |
| Mejora % | (RMSE_baseline - RMSE_GA) / RMSE_baseline | Maximizar |

---

## 5. Referencias

1. **Paper Base:**
   - Pramono, P. P., Surjandari, I., & Laoh, E. (2019). *Estimating Customer Segmentation based on Customer Lifetime Value Using Two-Stage Clustering Method*. IEEE ICSSSM 2019. [IEEE Xplore](https://ieeexplore.ieee.org/document/8887704/)

2. **LRFM y CLV:**
   - Alvandi, M., Fazli, S., & Abdoli, F. (2012). *K-Mean Clustering Method For Analysis Customer Lifetime Value With LRFM Relationship Model*. [Semantic Scholar](https://www.semanticscholar.org/paper/K-Mean-Clustering-Method-For-Analysis-Customer-With-Alvandi-Fazli/54ac58708a608bee195e9f60825a61da51ebbcf1)

3. **AHP para LRFM:**
   - *Measuring Customer Lifetime Value: Application of AHP in Determining Relative Weights of LRFM*. [IJAHP](https://www.ijahp.org/index.php/IJAHP/article/view/892)

4. **RFM Clásico:**
   - Khajvand, M., & Tarokh, M. J. (2011). *Estimating customer lifetime value based on RFM analysis*. [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S1877050910003868)

5. **Algoritmos Genéticos:**
   - Eiben, A. E., & Smith, J. E. (2015). *Introduction to Evolutionary Computing*. Springer.
