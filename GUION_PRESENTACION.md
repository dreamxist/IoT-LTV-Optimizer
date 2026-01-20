# Guion de Presentación
## Optimización Metaheurística de Modelos de Atribución Omnicanal en Entornos IoT

**Duración estimada:** 15-20 minutos

---

## DIAPOSITIVA 1: Título
**[Tiempo: 30 segundos]**

> Buenos días/tardes. Mi nombre es Francisco Zúñiga, estudiante de Ingeniería Civil Telemática de la Universidad Técnica Federico Santa María.
>
> Hoy les presentaré mi investigación sobre la optimización de modelos de atribución omnicanal utilizando algoritmos genéticos, enfocada en mejorar la predicción del Customer Lifetime Value en entornos con dispositivos IoT.

---

## DIAPOSITIVA 2: Problema de Investigación
**[Tiempo: 1.5 minutos]**

> Comencemos entendiendo el problema que aborda esta investigación.
>
> En el contexto actual del comercio minorista, los clientes interactúan con las empresas a través de múltiples canales: sitios web, aplicaciones móviles, email marketing, y cada vez más, a través de dispositivos IoT como beacons inteligentes, sistemas NFC y puntos de venta conectados.
>
> Esta proliferación de touchpoints crea un desafío significativo: **¿cómo atribuir correctamente el valor de cada canal a la conversión final del cliente?**
>
> Los modelos tradicionales presentan limitaciones importantes:
> - Los modelos basados en reglas, como "last-touch" o "first-touch", son arbitrarios y no reflejan la realidad del customer journey
> - Los métodos estadísticos asumen independencia entre canales, lo cual sabemos que no es cierto
> - Y los enfoques de machine learning, aunque precisos, carecen de interpretabilidad
>
> Esto nos lleva a nuestra pregunta de investigación: **¿Puede la computación evolutiva optimizar automáticamente los pesos de atribución para mejorar la precisión del CLV?**

---

## DIAPOSITIVA 3: Objetivos y Contribuciones
**[Tiempo: 1 minuto]**

> Nuestra investigación propone cuatro contribuciones principales:
>
> **Primero**, el desarrollo del framework OmniEvo, un algoritmo genético diseñado específicamente para optimizar pesos de atribución de forma automática, sin necesidad de juicio experto.
>
> **Segundo**, una validación experimental rigurosa que demuestra mejoras estadísticamente significativas sobre los métodos baseline.
>
> **Tercero**, un análisis de sensibilidad que evalúa la robustez del algoritmo ante variaciones en sus hiperparámetros.
>
> Y **cuarto**, insights específicos sobre la importancia de los canales IoT en la predicción del valor del cliente.
>
> Adelantando nuestro resultado principal: logramos una **mejora del 23% en RMSE** con significancia estadística confirmada.

---

## DIAPOSITIVA 4: Estado del Arte
**[Tiempo: 1.5 minutos]**

> Revisemos brevemente el estado del arte en tres áreas relevantes.
>
> En **modelos de CLV**, trabajos como el de Pramono utilizan clustering LRFM combinado con Fuzzy AHP para determinar pesos. Sin embargo, estos enfoques requieren juicio experto, no se adaptan automáticamente a los datos, y mantener consistencia en las matrices de comparación es difícil.
>
> En **modelos de atribución**, hemos visto una evolución desde heurísticas simples hasta modelos probabilísticos como Shapley. Pero estos presentan complejidad exponencial O(2^m), y los enfoques de machine learning sacrifican interpretabilidad.
>
> En **metaheurísticas aplicadas a marketing**, los algoritmos genéticos han sido exitosos en segmentación de clientes y optimización de campañas, pero su aplicación específica a problemas de atribución omnicanal está poco explorada.
>
> Esta brecha representa nuestra **oportunidad de investigación**: combinar la capacidad de búsqueda global de los algoritmos evolutivos con la necesidad de interpretabilidad en marketing.

---

## DIAPOSITIVA 5: Formulación del Problema
**[Tiempo: 1.5 minutos]**

> Formalicemos matemáticamente nuestro problema.
>
> Definimos **X** como una matriz de interacciones de dimensión n por m, donde cada elemento x_ij representa la intensidad de interacción del cliente i con el canal j.
>
> El vector **y** contiene los valores reales de CLV para cada cliente, y el vector **w** representa los pesos de atribución que queremos optimizar.
>
> Nuestra arquitectura considera **8 canales**: dos canales IoT (NFC y beacons), dos canales de aplicación (sesiones y wallet), y cuatro canales digitales tradicionales (email, web, ads y referidos).
>
> El problema de optimización busca **minimizar el RMSE** entre el CLV predicho (que es el producto Xw) y el CLV real.
>
> Las restricciones son fundamentales para la interpretabilidad:
> - La suma de los pesos debe ser igual a 1, lo que permite interpretarlos como proporciones o porcentajes
> - Y todos los pesos deben ser no negativos, ya que no tiene sentido que un canal contribuya negativamente al valor del cliente

---

## DIAPOSITIVA 6: Algoritmo Genético OmniEvo
**[Tiempo: 2 minutos]**

> Veamos los componentes de nuestro algoritmo genético.
>
> La **representación** utiliza vectores de números reales, donde cada gen corresponde al peso de un canal. Después de cada operación genética, normalizamos los vectores para mantener la restricción de suma unitaria.
>
> Para la **selección**, utilizamos torneo con k=3, lo que proporciona una presión selectiva balanceada que permite mantener diversidad sin perder convergencia.
>
> El **cruzamiento BLX-alpha** con alpha igual a 0.5 es particularmente importante. A diferencia del cruzamiento aritmético tradicional, BLX-alpha permite generar hijos fuera del rango definido por los padres, lo que facilita la exploración del espacio de búsqueda.
>
> La **mutación gaussiana** con sigma igual a 0.2 añade perturbaciones que ayudan a escapar de óptimos locales.
>
> El algoritmo completo, que pueden ver en el pseudocódigo, inicializa una población de 100 individuos, los evalúa usando el RMSE negativo como fitness, y evoluciona durante 50 generaciones aplicando selección, cruzamiento, mutación y elitismo.
>
> Implementamos todo usando **DEAP**, la biblioteca de algoritmos evolutivos en Python.

---

## DIAPOSITIVA 7: Configuración Experimental
**[Tiempo: 1 minuto]**

> Para la validación, generamos un **dataset sintético** de 1,000 clientes distribuidos en tres segmentos: alto valor (20%), medio valor (50%) y bajo valor (30%).
>
> Las interacciones en los 8 canales siguen distribuciones calibradas según la literatura de retail omnicanal, y el CLV incluye ruido gaussiano con sigma igual a 10 para simular condiciones realistas.
>
> El **protocolo de evaluación** divide los datos en 80% entrenamiento y 20% prueba de forma estratificada, y realizamos validación cruzada de 5 folds para estimar la varianza del modelo.
>
> Para **validación estadística**, aplicamos t-test pareado, prueba de Wilcoxon y calculamos la d de Cohen para medir el tamaño del efecto.
>
> Comparamos contra dos **baselines**: el modelo uniforme que asigna peso igual a todos los canales, y el modelo last-touch que asigna todo el crédito al último canal de interacción.

---

## DIAPOSITIVA 8: Resultados - Comparación de Rendimiento
**[Tiempo: 1.5 minutos]**

> Veamos los resultados de rendimiento.
>
> La tabla muestra que el modelo **last-touch** obtiene el peor desempeño con un RMSE de 45.23 y correlación de 0.612. Esto confirma que asignar todo el crédito al último touchpoint es una simplificación excesiva.
>
> El modelo **uniforme** mejora sustancialmente con RMSE de 31.17 y correlación de 0.789, lo que sugiere que considerar todos los canales es mejor que enfocarse en uno solo.
>
> Nuestro **algoritmo genético** logra el mejor rendimiento con RMSE de 23.99 y correlación de 0.874, representando una **mejora del 23%** sobre el baseline uniforme.
>
> La validación cruzada confirma estos resultados con un RMSE medio de 23.96 y una desviación estándar de solo 0.33, indicando **alta estabilidad**.
>
> En cuanto a significancia estadística, el t-test arroja un valor t de 45.32 con p menor a 0.0001, la prueba de Wilcoxon confirma con p menor a 0.0001, y la d de Cohen de 19.95 indica un **tamaño de efecto extremadamente grande**.

---

## DIAPOSITIVA 9: Resultados - Pesos de Atribución
**[Tiempo: 1.5 minutos]**

> Analicemos los pesos de atribución optimizados.
>
> El gráfico de barras muestra los pesos asignados a cada canal. Lo más notable es que los **canales IoT representan el 44.6%** del peso total de atribución.
>
> Los tres principales predictores del CLV son:
>
> **Primero**, las compras por NFC con un 35.7%. Esto tiene sentido porque las transacciones NFC representan un compromiso directo con la compra.
>
> **Segundo**, las recargas de wallet con 27.2%. Los clientes que mantienen saldo en su wallet digital demuestran intención de compra futura.
>
> **Tercero**, las aperturas de email con 22.9%, confirmando que el engagement con comunicaciones de marketing sigue siendo relevante.
>
> Es interesante notar que los canales tradicionales como web browsing y ads tienen pesos relativamente bajos, lo que sugiere que en este contexto omnicanal IoT, las **interacciones transaccionales son mejores predictores** que las interacciones de navegación.

---

## DIAPOSITIVA 10: Resultados - Análisis de Convergencia
**[Tiempo: 1 minuto]**

> El análisis de convergencia revela el comportamiento evolutivo del algoritmo.
>
> Como pueden observar en la curva, el **90% de la mejora se alcanza en las primeras 15 generaciones**, y la convergencia completa ocurre antes de la generación 30.
>
> Después de la convergencia, el fitness se mantiene estable sin degradación, lo que indica que el elitismo está funcionando correctamente.
>
> Este comportamiento demuestra un buen **balance entre exploración y explotación**:
> - El cruzamiento BLX-alpha permite explorar nuevas regiones del espacio de búsqueda
> - La mutación gaussiana ayuda a refinar las soluciones
> - Y el elitismo preserva las mejores soluciones encontradas
>
> Desde un punto de vista práctico, esto significa que **50 generaciones son suficientes** para encontrar una solución de alta calidad, lo que hace al algoritmo computacionalmente eficiente.

---

## DIAPOSITIVA 11: Resultados - Distribución de Errores
**[Tiempo: 1 minuto]**

> El boxplot compara la distribución de errores entre los tres modelos.
>
> El modelo **last-touch** en rojo muestra la mediana más alta (44.8) y la mayor dispersión, con un rango intercuartílico amplio. Esto indica predicciones inconsistentes y poco confiables.
>
> El modelo **uniforme** en amarillo reduce tanto la mediana (31.2) como la dispersión, pero aún presenta variabilidad considerable.
>
> Nuestro **algoritmo genético** en verde muestra la mediana más baja (24.2) y la dispersión más compacta. Esto significa que no solo predice mejor en promedio, sino que lo hace de forma **más consistente**.
>
> La reducción del 23% en RMSE, combinada con la menor varianza, confirma que la optimización evolutiva de los pesos de atribución produce un modelo superior tanto en precisión como en estabilidad.

---

## DIAPOSITIVA 12: Conclusiones y Trabajo Futuro
**[Tiempo: 1.5 minutos]**

> Resumiendo nuestras contribuciones principales:
>
> Desarrollamos el **framework OmniEvo** que optimiza automáticamente los pesos de atribución sin requerir juicio experto.
>
> Demostramos una **mejora del 23% en RMSE** con significancia estadística robusta (p menor a 0.0001).
>
> Identificamos que los **canales IoT son predictores críticos**, representando el 44.6% del peso total.
>
> Y confirmamos la **estabilidad del modelo** mediante validación cruzada con desviación estándar de solo 0.33.
>
> En cuanto a **limitaciones**, debemos reconocer que:
> - Los datos sintéticos, aunque calibrados, requieren validación con datos reales de retail
> - El modelo lineal asume que las contribuciones de los canales son aditivas
> - Y los pesos estáticos no capturan variaciones estacionales o temporales
>
> Como **trabajo futuro**, proponemos validar con datasets reales, incorporar kernels no lineales para capturar interacciones entre canales, implementar ventanas temporales para pesos dinámicos, y extender a un enfoque multi-objetivo que considere otras métricas además del RMSE.
>
> El código está disponible en GitHub para quienes deseen replicar o extender esta investigación.

---

## DIAPOSITIVA 13: Referencias y Cierre
**[Tiempo: 30 segundos]**

> Aquí pueden ver las principales referencias que fundamentan este trabajo, desde los modelos seminales de CLV de Fader y Hardie, hasta trabajos más recientes en atribución multicanal.
>
> En conclusión, esta investigación demuestra que la **computación evolutiva ofrece una alternativa viable y automática** a los métodos de atribución basados en heurísticas o juicio experto.
>
> Muchas gracias por su atención. Estoy disponible para responder sus preguntas.
>
> Mi correo es francisco.zunigap@usm.cl

---

## PREGUNTAS FRECUENTES (PREPARACIÓN)

### P: ¿Por qué usaron datos sintéticos en lugar de datos reales?
> Los datos de interacción omnicanal con CLV real son difíciles de obtener debido a restricciones de privacidad y confidencialidad comercial. Sin embargo, calibramos las distribuciones según literatura publicada de retail omnicanal, lo que nos permite demostrar la viabilidad del enfoque. La validación con datos reales es un paso natural de trabajo futuro.

### P: ¿Por qué un algoritmo genético y no otro método de optimización?
> Los algoritmos genéticos ofrecen varias ventajas para este problema: no requieren gradientes (el RMSE no es diferenciable en todos los puntos), mantienen diversidad en la búsqueda evitando óptimos locales, y los pesos resultantes son directamente interpretables. Comparado con otros métodos como PSO o evolución diferencial, los AG tienen mejor documentación en aplicaciones de marketing.

### P: ¿Cómo manejan el overfitting?
> Implementamos varias estrategias: validación cruzada de 5 folds para estimar la varianza del modelo, división estratificada para mantener la distribución de segmentos, y las restricciones de normalización actúan como una forma de regularización implícita. La baja desviación estándar (0.33) sugiere que el overfitting no es un problema significativo.

### P: ¿Es escalable a más canales?
> El algoritmo escala linealmente con el número de canales, ya que cada individuo tiene m genes (uno por canal). Para m=8 canales, la convergencia es rápida. Para escenarios con más canales, se podría aumentar el tamaño de la población proporcionalmente.

### P: ¿Qué limitaciones tiene el modelo lineal?
> El modelo lineal asume que las contribuciones de los canales son aditivas e independientes. En la realidad, pueden existir sinergias (email + beacon es más efectivo que la suma de ambos) o efectos de saturación. Incorporar kernels no lineales o términos de interacción es una extensión natural para capturar estos efectos.

---

**Tiempo total estimado: 15-18 minutos de presentación + 5-7 minutos de preguntas**
