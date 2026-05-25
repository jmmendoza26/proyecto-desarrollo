# Clasificación de Imágenes de Frutas y Vegetales: Transfer Learning con ResNet18 y CNN desde Cero

**Juan Manuel Mendoza**  
Asignatura: Deep Learning  
Universidad Católica de Colombia  
Mayo de 2026

---

## Resumen

Este informe documenta el desarrollo y comparación de dos enfoques de clasificación de imágenes para el reconocimiento automático de cinco categorías de frutas y vegetales: manzana (*Apple*), banana (*Banana*), tomate (*Tomato*), papa (*Potato*) y cebolla (*Onion*). La Parte 1 aplica transfer learning con ResNet18 preentrenado en ImageNet, descongelando la capa `layer4` y la capa clasificadora final (`fc`) para fine-tuning parcial. La Parte 2 implementa una CNN personalizada de cuatro bloques convolucionales entrenada desde cero, con `AdaptiveAvgPool2d` para eliminar la dependencia del tamaño de entrada. El dataset empleado fue Fruits-360 (100 imágenes por clase del dataset original más 10 imágenes propias por clase), con 450 muestras de entrenamiento y 100 de validación. Ambos modelos se entrenaron durante 20 épocas con el mismo DataLoader y las mismas transformaciones. ResNet18 alcanzó 99% de accuracy en validación controlada (máximo 100%) con macro F1-score de 0.99, frente al 76% de la CNN desde cero (máximo 87%, macro F1 de 0.74). La evaluación con 40 fotografías reales confirmó la ventaja del transfer learning: 70% para ResNet18 versus 48% para la CNN desde cero.

**Palabras clave:** reconocimiento de imágenes, redes neuronales convolucionales, transfer learning, ResNet18, fine-tuning, CNN desde cero, domain shift, Fruits-360

---

## 1. Introducción

El reconocimiento automático de imágenes de alimentos constituye un área de creciente interés en la intersección entre visión por computadora y aplicaciones prácticas en la industria alimentaria, la salud pública y el comercio minorista. La capacidad de identificar frutas y vegetales a partir de imágenes tiene aplicaciones directas en sistemas de inventario automatizado, aplicaciones de nutrición personal y control de calidad en cadenas de suministro.

El presente proyecto aborda este problema mediante el entrenamiento y comparación de dos clasificadores de imágenes basados en redes neuronales convolucionales (CNN). El primer enfoque aplica transfer learning sobre ResNet18 (He et al., 2016), una arquitectura con conexiones residuales preentrenada en ImageNet, descongelando parcialmente sus capas para fine-tuning. El segundo entrena una CNN diseñada desde cero, con el propósito de cuantificar el aporte del conocimiento preentrenado bajo condiciones controladas e idénticas.

El dataset empleado fue Fruits-360 (Muresan & Oltean, 2018), complementado con imágenes propias recolectadas en condiciones reales. La evaluación en dos contextos —imágenes controladas del dataset e imágenes reales no controladas— permite medir tanto el desempeño intrínseco de los modelos como su capacidad de generalización ante la brecha de dominio (*domain shift*).

---

## 2. Objetivos

### 2.1. Objetivo General

Desarrollar y comparar dos enfoques de clasificación de imágenes para el reconocimiento de cinco categorías de frutas y vegetales: transfer learning con fine-tuning parcial de ResNet18 y una CNN diseñada desde cero, evaluando el desempeño de ambos modelos en condiciones controladas y en fotografías reales.

### 2.2. Objetivos Específicos

- Implementar un pipeline completo de clasificación de imágenes en PyTorch con funciones reutilizables, abarcando desde la adquisición del dataset hasta la evaluación en condiciones controladas y reales.
- Aplicar transfer learning con fine-tuning parcial sobre ResNet18, descongelando `layer4` y la capa `fc`, y una CNN desde cero de cuatro bloques convolucionales, entrenando ambos modelos bajo exactamente las mismas condiciones.
- Evaluar y comparar el desempeño de ambos modelos mediante accuracy, F1-score por clase y evaluación con fotografías reales, cuantificando el impacto del conocimiento preentrenado y del fine-tuning en la generalización.

---

## 3. Metodología

La metodología está organizada en tres secciones: configuración y datos (compartida por ambos modelos), Parte 1 — transfer learning con ResNet18, y Parte 2 — CNN desde cero. Todas las secciones se ejecutan en un notebook de Python sobre Google Colab con aceleración GPU (CUDA). Las funciones de entrenamiento, evaluación y predicción se definen una sola vez y son reutilizadas por ambas partes, eliminando duplicación de código y garantizando condiciones experimentales idénticas.

### 3.1. Configuración del Entorno

Se instalaron y cargaron las bibliotecas necesarias: PyTorch como framework de aprendizaje profundo, Torchvision para modelos preentrenados y transformaciones, scikit-learn para métricas de evaluación, Matplotlib y PIL para visualización, y kagglehub para descarga automatizada del dataset. Se fijó una semilla aleatoria (SEED = 42) en Python, NumPy, PyTorch CPU y PyTorch CUDA para garantizar reproducibilidad. El dispositivo de ejecución confirmado fue GPU (CUDA).

### 3.2. Adquisición del Dataset

El dataset Fruits-360 (Muresan & Oltean, 2018) se descargó automáticamente desde Kaggle mediante la biblioteca kagglehub bajo el identificador `moltean/fruits`. Se trabajó con la versión de 100×100 píxeles (`fruits-360_100x100`), cuya carpeta `Training` contiene 260 subcarpetas. La carpeta fue localizada mediante una función de búsqueda recursiva que selecciona el directorio con mayor número de subdirectorios.

### 3.3. Definición de Clases y Selección de Imágenes

#### 3.3.1. Agrupación en Clases Generales

El dataset agrupa las frutas y vegetales en subclases muy específicas. Para este proyecto se consolidaron en cinco categorías generales usando coincidencia de prefijo en el nombre de la carpeta. El resultado fue el siguiente:

- **Apple:** 30 subclases, 16.571 imágenes disponibles
- **Tomato:** 19 subclases, 10.259 imágenes disponibles
- **Onion:** 6 subclases, 3.128 imágenes disponibles
- **Banana:** 5 subclases, 1.917 imágenes disponibles
- **Potato:** 4 subclases, 1.803 imágenes disponibles

#### 3.3.2. Selección y División del Conjunto de Datos

Se seleccionaron aleatoriamente 100 imágenes por clase del dataset Fruits-360, distribuidas en 80 para entrenamiento y 20 para validación. Adicionalmente, se incorporaron 10 imágenes propias por clase (50 en total) al conjunto de entrenamiento, provenientes de un dataset personal almacenado en Google Drive (`dataset_frutas_vegetales`). Estas imágenes, tomadas en condiciones reales, aumentan la diversidad del entrenamiento y contribuyen a reducir la brecha de dominio. El conjunto final fue de 450 muestras de entrenamiento (90 por clase, incluyendo las 10 propias) y 100 de validación (20 por clase). Las imágenes no fueron copiadas; el notebook conserva únicamente las rutas de archivo originales.

### 3.4. Preprocesamiento e Ingeniería de Datos

Las transformaciones se definen una sola vez y se reutilizan en los DataLoaders de entrenamiento y validación, en la función de predicción individual y en la función de evaluación con fotos reales, garantizando preprocesamiento idéntico en todos los contextos.

**Para el conjunto de entrenamiento:** (1) `Resize(224, 224)`; (2) `RandomHorizontalFlip()`; (3) `RandomRotation(±20°)`: rotación para compensar ángulos no frontales; (4) `ColorJitter(brightness=0.2, contrast=0.2)`: variación moderada de brillo y contraste para simular condiciones de iluminación variables; (5) `ToTensor()`; (6) normalización con parámetros oficiales de `ResNet18_Weights.DEFAULT`.

**Para el conjunto de validación:** `Resize(224, 224)`, `ToTensor()` y la misma normalización ImageNet. Sin transformaciones aleatorias para evaluación determinista.

Los DataLoaders se configuraron con `batch_size = 32`, `num_workers = 2` y `pin_memory = True`, resultando en 15 batches por época en entrenamiento.

### 3.5. Funciones Reutilizables

El notebook define cuatro funciones centrales compartidas por ambos modelos:

- **`train_model`:** ejecuta el bucle completo de entrenamiento (forward, loss, backward, optimizer step, validación por época) y retorna el historial de métricas. Acepta un scheduler opcional.
- **`evaluate_model`:** calcula accuracy total, genera la matriz de confusión y el reporte de clasificación completo sobre un DataLoader.
- **`predict_image`:** predice la clase de una imagen individual y muestra la imagen con la clase predicha y la confianza (softmax).
- **`evaluar_fotos_reales`:** evalúa el modelo sobre fotografías organizadas en carpetas por clase en Google Drive, genera una grilla visual de predicciones y reporta accuracy por clase.

### 3.6. Parte 1 — Transfer Learning con ResNet18

#### 3.6.1. Arquitectura y Estrategia de Fine-Tuning

Se cargó ResNet18 con los pesos preentrenados de ImageNet (`ResNet18_Weights.DEFAULT`). Se aplicó fine-tuning parcial: se congelaron todas las capas excepto `layer4` (el bloque convolucional de más alto nivel) y la capa clasificadora `fc`. La capa `fc` original fue reemplazada por `Linear(512 → 5)`. El modelo resultante tiene 11.179.077 parámetros totales y 8.396.293 entrenables (75% del modelo). Descongelar `layer4` permite que las representaciones de alto nivel se adapten al dominio visual de frutas y vegetales sin modificar las características de bajo nivel aprendidas en ImageNet.

#### 3.6.2. Entrenamiento

Se utilizaron dos grupos de parámetros con tasas de aprendizaje diferenciadas: `layer4` con lr = 1e-4 (conservador) y `fc` con lr = 1e-3 (estándar). El optimizador fue Adam y la función de pérdida CrossEntropyLoss. Se aplicó un scheduler `StepLR` que reduce el lr a la mitad (gamma = 0.5) cada 5 épocas. El entrenamiento duró 20 épocas.

### 3.7. Parte 2 — CNN desde Cero

#### 3.7.1. Arquitectura

Se diseñó la clase `FruitVegetableCNN` con cuatro bloques convolucionales. Cada bloque contiene: `Conv2d` (kernel 3×3, padding 1), `BatchNorm2d`, `ReLU` y `MaxPool2d(2)`. El número de filtros aumenta progresivamente (3 → 32 → 64 → 128 → 256). Tras los cuatro bloques, `AdaptiveAvgPool2d(1)` reduce cada mapa de características a un único valor por canal, eliminando la necesidad de calcular dimensiones fijas. El clasificador consta de `Flatten`, `Linear(256 → 128)`, `ReLU`, `Dropout(0.4)` y `Linear(128 → 5)`. La arquitectura tiene 422.917 parámetros, todos entrenables. Se verificaron las dimensiones con un forward pass sobre una imagen dummy antes del entrenamiento.

#### 3.7.2. Entrenamiento

El optimizador Adam con lr = 1e-3 actualiza todos los parámetros. Se usó el mismo `StepLR` (step_size = 5, gamma = 0.5), CrossEntropyLoss y 20 épocas que en la Parte 1, garantizando una comparación justa.

---

## 4. Resultados Obtenidos

### 4.1. ResNet18 — Evolución del Entrenamiento

El modelo convergió de forma muy rápida: en la tercera época ya alcanzó 99% de accuracy en validación, y en las épocas 6 y 8 alcanzó el 100%. La pérdida descendió de 0.8257 a 0.0174.

*Tabla 1. Evolución de pérdida y accuracy durante las 20 épocas de entrenamiento de ResNet18*

| Época | Pérdida | Acc. Entrenamiento | Acc. Validación |
|------:|--------:|------------------:|----------------:|
| 1  | 0.8257 | 73.11% | 91.00% |
| 2  | 0.1920 | 95.56% | 98.00% |
| 3  | 0.0814 | 98.44% | 99.00% |
| 4  | 0.0673 | 98.44% | 99.00% |
| 5  | 0.1008 | 96.89% | 98.00% |
| 6  | 0.0494 | 99.33% | 100.00% |
| 7  | 0.0289 | 99.33% | 99.00% |
| 8  | 0.0443 | 99.33% | 100.00% |
| 9  | 0.0409 | 98.89% | 99.00% |
| 10 | 0.0400 | 98.67% | 98.00% |
| 11 | 0.0258 | 99.78% | 98.00% |
| 12 | 0.0303 | 99.56% | 99.00% |
| 13 | 0.0394 | 98.89% | 99.00% |
| 14 | 0.0206 | 99.56% | 98.00% |
| 15 | 0.0355 | 99.11% | 99.00% |
| 16 | 0.0303 | 99.11% | 99.00% |
| 17 | 0.0250 | 99.56% | 99.00% |
| 18 | 0.0256 | 99.78% | 99.00% |
| 19 | 0.0218 | 99.56% | 99.00% |
| 20 | 0.0174 | 99.78% | 99.00% |

La convergencia acelerada es consecuencia directa del fine-tuning parcial con pesos de ImageNet: el modelo disponía de representaciones visuales robustas y solo necesitó ajustar las capas de mayor nivel. La brecha entre accuracy de entrenamiento (99.78%) y validación (99.00%) en la época final es mínima, indicando generalización excelente dentro del dominio controlado.

### 4.2. ResNet18 — Reporte de Clasificación

El accuracy final en validación controlada fue **0.9900** (99 de 100 imágenes correctas).

*Tabla 2. Reporte de clasificación de ResNet18 sobre el conjunto de validación controlada (n = 100)*

| Clase     | Precision | Recall | F1-Score | Soporte |
|:----------|----------:|-------:|---------:|--------:|
| Apple     | 1.00 | 1.00 | 1.00 | 20 |
| Banana    | 1.00 | 1.00 | 1.00 | 20 |
| Tomato    | 0.95 | 1.00 | 0.98 | 20 |
| Potato    | 1.00 | 0.95 | 0.97 | 20 |
| Onion     | 1.00 | 1.00 | 1.00 | 20 |
| **Macro avg** | **0.99** | **0.99** | **0.99** | **100** |

Los resultados muestran desempeño casi perfecto en todas las clases. Apple, Banana y Onion obtuvieron F1-score de 1.00. Tomato (F1 = 0.98) y Potato (F1 = 0.97) presentaron el único error cada uno. El macro F1-score de 0.99 confirma generalización homogénea sin sesgos hacia clases particulares.

### 4.3. ResNet18 — Predicción Individual

La prueba de predicción individual se realizó sobre una imagen aleatoria de la clase Banana. ResNet18 predijo correctamente la clase con una confianza del **99.93%**, valor que refleja la alta certeza de las representaciones aprendidas con fine-tuning parcial.

### 4.4. ResNet18 — Evaluación con Fotografías Reales

*Tabla 3. Resultados de la evaluación de ResNet18 con fotografías reales (n = 40)*

| Clase  | Correctas | Total | Accuracy | Observación |
|:-------|----------:|------:|---------:|:------------|
| Apple  | 9  | 12 | 75%  | Buena generalización |
| Banana | 7  | 7  | 100% | Forma muy distintiva |
| Tomato | 8  | 8  | 100% | Reconocimiento perfecto |
| Potato | 0  | 6  | 0%   | Alta sensibilidad al contexto |
| Onion  | 4  | 7  | 57%  | Confusión en algunas variedades |
| **Total** | **28** | **40** | **70%** | |

Banana y Tomato obtuvieron rendimiento perfecto (100%). Potato fue la clase más problemática con 0% de aciertos, lo que sugiere que sus representaciones visuales en fotografías reales difieren sustancialmente de las imágenes de fondo blanco del dataset.

### 4.5. CNN desde Cero — Evolución del Entrenamiento

A diferencia de ResNet18, la convergencia fue lenta e inestable, con fluctuaciones pronunciadas en el accuracy de validación entre épocas consecutivas. La pérdida descendió de 1.4190 a 0.6388, a un ritmo notablemente más lento.

*Tabla 4. Evolución de pérdida y accuracy durante las 20 épocas de entrenamiento de la CNN desde cero*

| Época | Pérdida | Acc. Entrenamiento | Acc. Validación |
|------:|--------:|------------------:|----------------:|
| 1  | 1.4190 | 42.00% | 45.00% |
| 2  | 1.1673 | 53.56% | 44.00% |
| 3  | 1.0206 | 61.78% | 70.00% |
| 4  | 0.9831 | 59.33% | 56.00% |
| 5  | 0.8955 | 66.67% | 47.00% |
| 6  | 0.8925 | 66.22% | 51.00% |
| 7  | 0.8291 | 68.44% | 85.00% |
| 8  | 0.8166 | 69.56% | 76.00% |
| 9  | 0.8195 | 67.56% | 81.00% |
| 10 | 0.8038 | 71.11% | 75.00% |
| 11 | 0.7334 | 73.33% | 86.00% |
| 12 | 0.7369 | 75.56% | 82.00% |
| 13 | 0.7276 | 74.44% | 80.00% |
| 14 | 0.7004 | 76.00% | 79.00% |
| 15 | 0.7122 | 74.67% | 85.00% |
| 16 | 0.6678 | 78.22% | 80.00% |
| 17 | 0.6589 | 76.44% | 87.00% |
| 18 | 0.6407 | 77.33% | 85.00% |
| 19 | 0.6457 | 78.44% | 80.00% |
| 20 | 0.6388 | 78.89% | 76.00% |

El accuracy máximo de validación fue del 87% (época 17), pero el valor de la última época retrocedió al 76%, indicando que el modelo no logró una convergencia estable con los datos disponibles.

### 4.6. CNN desde Cero — Reporte de Clasificación

El accuracy final en validación fue **0.7600** (76 de 100 imágenes correctas).

*Tabla 5. Reporte de clasificación de la CNN desde cero sobre el conjunto de validación controlada (n = 100)*

| Clase     | Precision | Recall | F1-Score | Soporte |
|:----------|----------:|-------:|---------:|--------:|
| Apple     | 1.00 | 0.35 | 0.52 | 20 |
| Banana    | 0.65 | 1.00 | 0.78 | 20 |
| Tomato    | 0.81 | 0.85 | 0.83 | 20 |
| Potato    | 0.74 | 0.85 | 0.79 | 20 |
| Onion     | 0.83 | 0.75 | 0.79 | 20 |
| **Macro avg** | **0.81** | **0.76** | **0.74** | **100** |

Apple fue la clase más problemática con F1-score de 0.52, consecuencia de un recall extremadamente bajo (0.35): el modelo identificó correctamente solo 7 de las 20 imágenes de manzana. La precisión perfecta de Apple (1.00) indica que cuando el modelo predijo Apple lo hizo correctamente, pero la mayoría de las manzanas fueron clasificadas como otras clases.

### 4.7. CNN desde Cero — Evaluación con Fotografías Reales

La evaluación con las mismas 40 fotografías reales arrojó un accuracy total del **48%** para la CNN desde cero: Apple 58% (7/12), Tomato 62% (5/8), Banana 57% (4/7), Onion 43% (3/7) y Potato 0% (0/6).

### 4.8. Comparación entre Enfoques

*Tabla 6. Comparación de resultados entre ResNet18 (transfer learning) y CNN desde cero*

| Métrica                              | ResNet18 (Transfer Learning) | CNN desde Cero |
|:-------------------------------------|-----------------------------:|---------------:|
| Accuracy validación (época 20)       | 99.00%       | 76.00%     |
| Accuracy validación (máximo)         | 100.00%      | 87.00%     |
| Macro F1-score validación            | 0.99         | 0.74       |
| Accuracy fotos reales (40 imgs)      | 70%          | 48%        |
| Parámetros totales                   | 11,179,077   | 422,917    |
| Parámetros entrenables               | 8,396,293    | 422,917    |
| Pesos iniciales                      | ImageNet     | Aleatorios |
| Capas descongeladas                  | layer4 + fc  | Todas      |
| Épocas entrenadas                    | 20           | 20         |

ResNet18 superó a la CNN desde cero en todos los escenarios. La rapidez de convergencia de ResNet18 (99% en la tercera época) frente a la lenta y volátil curva de la CNN (fluctuaciones de hasta 30 puntos porcentuales entre épocas consecutivas en las primeras iteraciones) ilustra cuantitativamente el valor del transfer learning cuando los datos disponibles son limitados. Un resultado destacable es que la CNN desde cero, con solo 422.917 parámetros (26 veces menos que ResNet18), compitió razonablemente en el dominio controlado, pero no logró generalizar con la misma eficacia ante condiciones reales.

---

## 5. Limitaciones y Mejoras Futuras

### 5.1. Limitaciones del Enfoque Implementado

**Brecha de dominio persistente.** A pesar de incorporar 10 imágenes propias por clase al entrenamiento, Potato obtuvo 0% de accuracy en fotografías reales en ambos modelos. Esto indica que la distribución visual de esta clase en condiciones reales difiere fundamentalmente de las imágenes controladas de Fruits-360, y que 10 imágenes propias por clase son insuficientes para cubrir esa variabilidad.

**Validación con imágenes controladas únicamente.** El conjunto de validación proviene exclusivamente de Fruits-360 (imágenes con fondo blanco uniforme), lo que genera métricas optimistas que no reflejan el desempeño real. La evaluación con fotografías reales ofrece una medida más representativa, aunque su muestra (40 imágenes) sigue siendo estadísticamente reducida.

**Inestabilidad de la CNN desde cero.** La CNN desde cero no convergió de forma estable en 20 épocas, con fluctuaciones de hasta 39 puntos porcentuales en el accuracy de validación entre épocas consecutivas (épocas 6 y 7: 51% → 85%). Esto indica que 450 imágenes de entrenamiento son insuficientes para optimizar de forma confiable 422.917 parámetros inicializados aleatoriamente.

**Dataset de evaluación real pequeño y no balanceado.** Las 40 fotografías reales están distribuidas de forma desigual entre clases (12 de Apple, 6 de Potato), lo que hace que el accuracy total esté influenciado por el desempeño en las clases con más imágenes.

### 5.2. Mejoras Futuras

- Ampliar el dataset propio a 50–100 imágenes reales por clase, con especial énfasis en Potato y Onion, que mostraron mayor degradación en la evaluación real, para reducir la brecha de dominio sin depender exclusivamente de Fruits-360.
- Implementar early stopping basado en el accuracy de validación, deteniendo el entrenamiento cuando el modelo deja de mejorar durante un número definido de épocas, lo que evitaría el retroceso observado en la CNN desde cero entre las épocas 17 y 20.
- Construir un conjunto de validación real balanceado con al menos 20 imágenes por clase tomadas en condiciones no controladas, de modo que las métricas de validación reflejen mejor el desempeño en condiciones reales de uso.

---

## Referencias

He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 770–778. https://doi.org/10.1109/CVPR.2016.90

Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic optimization. *Proceedings of the 3rd International Conference on Learning Representations (ICLR)*. https://arxiv.org/abs/1412.6980

Muresan, H., & Oltean, M. (2018). Fruit recognition from images using deep learning. *Acta Universitatis Sapientiae, Informatica*, *10*(1), 26–42. https://doi.org/10.2478/ausi-2018-0002

Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., Antiga, L., Desmaison, A., Kopf, A., Yang, E., DeVito, Z., Raison, M., Tejani, A., Chilamkurthy, S., Steiner, B., Fang, L., … Chintala, S. (2019). PyTorch: An imperative style, high-performance deep learning library. *Advances in Neural Information Processing Systems*, *32*, 8024–8035. https://papers.nips.cc/paper/2019/hash/bdbca288fee7f92f2bfa9f7012727740-Abstract.html

Russakovsky, O., Deng, J., Su, H., Krause, J., Satheesh, S., Ma, S., Huang, Z., Karpathy, A., Khosla, A., Bernstein, M., Berg, A. C., & Fei-Fei, L. (2015). ImageNet large scale visual recognition challenge. *International Journal of Computer Vision*, *115*(3), 211–252. https://doi.org/10.1007/s11263-015-0816-y
