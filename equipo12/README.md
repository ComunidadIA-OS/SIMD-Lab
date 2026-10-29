# Documentación de Arquitectura — Equipo SIMD LBb

Trabajo desarrollado por Celia Almendros Saelices, Esther Almendros Saelices y Teresa Valero Díaz.

## 1. Propósito de la Plataforma, Métricas y Fórmulas de Ponderación

### Objetivo del Sistema

La plataforma está diseñada para auditar la presencia, relevancia y tratamiento de género en los contenidos multimedia que publican o asocian a empresas e instituciones. Evalúa en qué medida se da voz, visibilidad y liderazgo a las mujeres en cuatro formatos: **texto**, **audio**, **imagen** y **vídeo**, asegurando un análisis alineado con criterios de sostenibilidad social y gobernanza (ESG) y el reglamento europeo de inteligencia artificial (**EU AI Act**).

---

### Métricas Analizadas por Modalidad

1. **Texto (9 métricas):**
* *Proporción de frases*: porcentaje de frases referidas a mujeres (`sentence_female_ratio`), hombres (`sentence_male_ratio`) y neutras (`sentence_neutral_ratio`).
* *Diversidad léxica y sufijos*: variedad de raíces léxicas femeninas (`lexical_root_diversity`) y sufijos que marcan género (`feminine_suffix_ratio`).
* *Frecuencia de mención*: recuento absoluto y frecuencia relativa frente al total del texto (`relative_frequency_score`).
* *Agencia sintáctica*: mide si la mujer aparece como sujeto activo de las acciones y verbos (`syntactic_agency_ratio`).
* *Voz experta y titular*: frecuencia con la que se citan declaraciones directas como especialista (`expert_quote_ratio`) y presencia en titulares (`headline_presence`).


2. **Audio (7 métricas):**
* *Apertura y cierre*: si una voz femenina abre (`female_opened_score`) o despide (`female_closed_score`) la locución o programa.
* *Tiempo y turnos de palabra*: porcentaje de tiempo de voz femenina (`female_speaking_time_ratio`) y proporción de turnos tomados (`speaking_turns_ratio`).
* *Presencia en el tramo central*: participación entre el 20% y el 80% de la grabación (`middle_segment_female_ratio`).
* *Fiabilidad acústica y no interrupción*: certeza del modelo sobre el género de la voz (`acoustic_gender_confidence`) y fluidez sin ser pisada o interrumpida (`no_interruption_score`).


3. **Imagen (8 métricas):**
* *Presencia visual*: ratio de rostros y cuerpos femeninos (`visual_gender_female_ratio`).
* *Composición y lenguaje visual*: tipo de plano de importancia (`shot_scale_importance_score`), ángulo de cámara que denote autoridad (`camera_angle_power_score`), ubicación según la regla de los tercios (`rule_of_thirds_alignment`) e iluminación/sombras (`shadow_lighting_prominence`).
* *Centralidad y contexto*: posición centrada en la composición (`spatial_centrality_score`), persistencia entre fotos de la misma galería (`cross_image_persistence_ratio`) y contexto visual en puestos de liderazgo (`leadership_visual_context`).


4. **Vídeo (5 métricas):**
* *Pista de audio y fotogramas*: hereda y evalúa tanto la pista de audio como las capturas visuales.
* *Tiempo en pantalla*: porcentaje del tiempo total que aparece una mujer en cámara (`female_screen_time_ratio`).
* *Dinamismo y sincronía*: movimiento mediante flujo óptico (`optical_flow_motion_score`), dominancia cinética (`kinetic_dominance_ratio`), cambios en silueta (`silhouette_change_histogram_score`) y sincronización entre quien habla y la imagen activa (`active_speaker_sync_score`).



---

### Fórmulas de Cálculo y Ponderación

El sistema asigna una puntuación entre **0 y 100** a cada modalidad. Si en alguna métrica no hay datos (por ejemplo, si nadie interrumpe en un audio o no hay citas directas en un texto), **no se penaliza asignando un cero**, sino que ese indicador se descarta y su peso se redistribuye proporcionalmente entre los que sí tienen valor medido:

$$\text{Puntuación}(\text{modalidad}) = \frac{\sum_{i \in \text{Válidos}} (w_i \cdot v_i)}{\sum_{i \in \text{Válidos}} w_i} \times 100$$

Donde $w_i$ es el peso asignado al indicador y $v_i \in [0, 1]$ es el valor de la métrica.

#### Ponderación Interna por Modalidad:

* **Texto:**

$$\text{Score}_{\text{texto}} = 0.18(\text{frases fem.}) + 0.04(\text{frases masc.}) + 0.06(\text{neutras}) + 0.08(\text{diversidad}) + 0.08(\text{sufijos}) + 0.10(\text{frecuencia}) + 0.16(\text{agencia}) + 0.15(\text{citas experta}) + 0.07(\text{titular}) + 0.08(\text{menciones norm.})$$


* **Audio:**

$$\text{Score}_{\text{audio}} = 0.25(\text{tiempo voz fem.}) + 0.15(\text{turnos habla}) + 0.12(\text{segmento central}) + 0.10(\text{apertura}) + 0.10(\text{cierre}) + 0.10(\text{no interrupción}) + 0.08(\text{confianza acústica}) + 0.10(\text{conteo turnos})$$


* **Imagen:**

$$\text{Score}_{\text{imagen}} = 0.20(\text{ratio visual fem.}) + 0.14(\text{contexto liderazgo}) + 0.10(\text{escala de plano}) + 0.10(\text{ángulo cámara}) + 0.10(\text{persistencia}) + 0.10(\text{centralidad}) + 0.08(\text{tercios}) + 0.08(\text{iluminación}) + 0.10(\text{apariciones})$$


* **Vídeo:**

$$\text{Score}_{\text{video}} = 0.35(\text{tiempo en pantalla}) + 0.20(\text{sincronía de habla}) + 0.15(\text{flujo óptico}) + 0.15(\text{dominancia cinética}) + 0.15(\text{cambio silueta})$$



#### Puntuación por Documento (`importance_score`):

Para cada documento individual, se promedian las puntuaciones de las modalidades que contiene. Si un documento no presenta ninguna modalidad analizable directamente, se recurre a la predicción base del modelo **LightGBM** (entrenado con restricciones monótonas que penalizan el sesgo):

$$\text{Puntuación Documento} = \begin{cases} \text{Media}(\text{Modalidades Válidas del Documento}) & \text{si hay modalidades presentes} \\ \text{Predicción Base LightGBM} & \text{si no hay métricas directas} \end{cases}$$

#### Índice Global Ponderado de la Entidad:

El **Índice Global Ponderado** (el indicador central mostrado en la interfaz) consolida las cuatro modalidades ponderando cada una según su **volumen de presencia** (el número de documentos válidos de ese canal, $N_m$):

$$\text{Índice Global Ponderado} = \frac{\sum_{m \in \{\text{Texto, Audio, Imagen, Vídeo}\}} N_m \cdot \text{Score}_m}{\sum_{m} N_m}$$

De este modo, si la auditoría ha analizado mayoritariamente artículos de prensa, el canal de texto tendrá un peso proporcionalmente mayor en la nota final, coincidiendo de forma exacta con los porcentajes de peso visualizados en la interfaz gráfica.

---

## 2. Fuentes de Datos y Conexión con Servicios Externos

La plataforma recopila información a través de clientes especializados en `src/connectors/`:

1. **The Guardian (`guardian_client.py`):** Conecta con la Open Platform de The Guardian para obtener noticias, artículos de opinión, autores, textos íntegros y enlaces a fotografías de prensa.
2. **YouTube (`youtube_client.py`):** Extrae metadatos de canales oficiales, fragmentos de vídeo de ruedas de prensa y debates mediante la API de YouTube y utilidades de descarga.
3. **Wikimedia Commons (`wikimedia_client.py`):** Recupera fotografías de directivos, consejos de administración y eventos corporativos con licencias abiertas y metadatos estructurados.
4. **Internet Archive (`internet_archive_client.py`):** Consulta grabaciones sonoras históricas, vídeos de archivo y hemeroteca digital.
5. **Podcast Index (`podcastindex_client.py`):** Busca entrevistas y episodios de podcast asociados a la organización analizada.
6. **Flickr (`flickr_client.py`):** Permite complementar la búsqueda de imágenes bajo licencias Creative Commons.

---

## 3. Extracción de Métricas, Modelos y Procesamiento Previo

Antes de extraer métricas definitivas, los datos pasan por varias etapas de acondicionamiento:

### Procesamiento Previo y Protección de Privacidad

* **Limpieza de Privacidad (`privacy_scrubber.py`):** Las transcripciones y textos se limpian de datos de carácter personal (PII) mediante reglas de anonimización (sustituyendo identificadores por etiquetas anónimas). Además, los archivos multimedia descargados se guardan temporalmente en carpetas efímeras y se destruyen al finalizar la sesión.
* **Recorte multimedia con FFmpeg:** El audio se normaliza a formato mono de 16 kHz para los modelos de voz, y los vídeos se segmentan extrayendo fotogramas representativos a 1 imagen por segundo (1 fps).

### Modelos de Inteligencia Artificial Utilizados

* **Texto (`TextGenderAnalyzer`):**
* Emplea modelos lingüísticos de **spaCy** (`es_core_news_md` y `en_core_web_md`) para el análisis sintáctico de dependencias (para determinar quién es el sujeto activo del verbo), etiquetado morfológico y concordancia léxica apoyado por un vocabulario de género específico (`vocabulario_genero.py`).


* **Audio (`AudioGenderAnalyzer`):**
* Utiliza **inaSpeechSegmenter**, una red neuronal convolucional entrenada para segmentar señales de audio en tramos de música, silencio y habla diferenciada entre hombre y mujer.


* **Imagen y Vídeo (`VisionGenderAnalyzer`):**
* Utiliza **YuNet** (detección facial ligera y rápida) para localizar rostros en imágenes y fotogramas.
* Utiliza **OpenCLIP** (redes multimodales de contraste visión-lenguaje) para clasificar atributos de contexto, puestos de responsabilidad y composición de la imagen (ángulos, planos y entorno).



---

## 4. Estructura del Modelo de Datos (Schemas)

El proyecto utiliza **Pydantic** (`src/common/schemas.py`) para validar estrictamente la información entre fases:

* **`UnifiedMediaDocument`:** Representa un documento tras la descarga de la fuente. Contiene:
* Identificador único (`doc_id`), fuente (`source`), término buscado (`company_query`), URL y título.
* Texto en bruto (`raw_text`), fecha de publicación, autor e idioma normalizado en formato ISO 639-1.
* Listas de enlaces remotos (`image_urls`, `audio_urls`, `video_urls`) y rutas locales a los archivos temporales (`image_paths`, `audio_paths`, `video_paths`, `frame_dirs`).
* Modalidades disponibles (`available_modalities`).


* **Métricas Especializadas:** Clases `TextMetrics`, `AudioMetrics`, `ImageMetrics` y `VideoMetrics` que agrupan los valores numéricos calculados (todos entre 0.0 y 1.0).
* **`ExtractedFeaturesPayload`:** Objeto central consolidado. Reúne las métricas de las 4 modalidades calculadas para ese documento y ofrece métodos para convertir los datos en una fila tabular compatible con modelos tabulares y análisis explicativo.

---

## 5. Pantallas de la Interfaz Gráfica (Streamlit)

La aplicación web (`frontend/app.py`) ofrece un entorno visual estructurado en una barra lateral y tres pestañas temáticas:

1. **Barra Lateral (`search_sidebar.py`):**
* Permite introducir el nombre de la empresa a auditar, activar o desactivar fuentes de datos individuales, configurar el límite de documentos y alternar entre modo real o modo de prueba rápida.


2. **Pestaña 1: Panel ESG (`tab_esg_dashboard.py`):**
* Muestra el Índice Global Ponderado de la entidad auditada, los pesos relativos de cada canal, desgloses por fuente, tarjetas con indicadores clave de paridad y gráficos comparativos de participación.


3. **Pestaña 2: Cumplimiento Normativo y Equidad (`tab_ai_act_fairness.py`):**
* Resume el nivel de riesgo de acuerdo al marco del EU AI Act, lista posibles sesgos detectados y muestra las alertas de desequilibrio representativo.


4. **Pestaña 3: Explicabilidad con SHAP y LIME (`tab_xai_shap_lime.py`):**
* Presenta gráficos de impacto para entender por qué la puntuación subió o bajó, indicando el peso relativo que ha tenido cada métrica en la evaluación final.



---

## 6. Flujo del Sistema al Pulsar "Auditar"

Cuando el usuario introduce el término de búsqueda y hace clic en el botón de auditoría en Streamlit, el orden de ejecución entre componentes es el siguiente:

```text
[Usuario pulsa 'Auditar' en Streamlit]
                  │
                  ▼
         main_controller.py
                  │
                  ├─► 1. Clientes de Ingesta (Guardian, YouTube, etc.)
                  │      Descargan metadatos y enlaces de la empresa.
                  │
                  ├─► 2. semantic_ranker.py
                  │      Compara embeddings de texto y selecciona los documentos más relevantes.
                  │
                  ├─► 3. media_downloader.py
                  │      Descarga los archivos multimedia de los documentos seleccionados.
                  │
                  ├─► 4. multimodal_pipeline.py
                  │      ├─► privacy_scrubber.py (anonimiza datos personales)
                  │      ├─► text_analyzer.py    (extrae métricas de texto con spaCy)
                  │      ├─► audio_analyzer.py   (segmenta voces con inaSpeechSegmenter)
                  │      └─► vision_analyzer.py  (evalúa planos y rostros con YuNet y OpenCLIP)
                  │
                  ├─► 5. scoring_model.py
                  │      Calcula las medias por modalidad, los scores por documento y 
                  │      el Índice Global Ponderado según el peso de cada canal.
                  │
                  ├─► 6. shap_explainer.py / lime_explainer.py
                  │      Calculan los pesos de atribución para cada documento.
                  │
                  ├─► 7. audit_logger.py
                  │      Registra la huella de ejecución y el gasto energético en los logs.
                  │
                  └─► 8. Retorno a app.py
                         Renderiza los resultados, gráficos y alertas en pantalla.

```

---

## 7. Explicabilidad, Supervisión Humana y Cumplimiento Normativo

### Explicabilidad (XAI)

* **SHAP (`shap_explainer.py`):** Utiliza *TreeExplainer* sobre el modelo LightGBM para descomponer la puntuación. Permite visualizar qué variables empujaron la nota hacia arriba (por ejemplo, alta presencia en titulares o mayor tiempo de voz) y cuáles restaron puntos.
* **LIME (`lime_explainer.py`):** Genera explicaciones locales perturbando pequeñas características para comprobar la estabilidad de la predicción en un documento individual.
* **Evidencia Textual (`text_evidence.py`):** Localiza y subraya en el texto las frases concretas y citas que originaron las puntuaciones de liderazgo o agencia.

### Supervisión Humana (*Human-in-the-Loop*)

* A través de `human_oversight.py`, la plataforma no actúa como un sistema cerrado. Un auditor humano puede revisar las puntuaciones automáticas en la tabla interactiva de la aplicación, editar valores anómalos o justificar discrepancias.
* Al aplicar cambios manuales, el sistema recalcula la puntuación global, almacena un registro de diferencias con el recuento de filas modificadas y añade una marca de tiempo que garantiza la validez de la revisión.

### Cumplimiento del EU AI Act

* **Trazabilidad y Registro de Auditoría (`audit_logger.py`):** Toda decisión, parámetro de entrada y resultado queda guardado de manera estructurada con sellos de tiempo inmutables.
* **Sostenibilidad y Huella Ambiental:** Utiliza la librería **CodeCarbon** para registrar en tiempo real el consumo de CPU, memoria RAM y estimar los gramos de $\text{CO}_2$ equivalentes producidos durante la auditoría, dando cumplimiento al Anexo IV y Artículo 11 del marco europeo sobre el impacto ambiental de sistemas de inteligencia artificial.

---

## 8. Pasos para crear el entorno virtual (Python 3.11, 3.12 o 3.13)
Abre una terminal en la carpeta raíz del repositorio (SIMD-Lab) y ejecuta los siguientes comandos en orden:

pip install --user virtualenv

python3 -m virtualenv .venv

source .venv/bin/activate

pip install -r equipo12/requirements/requirements.txt

pip install inaSpeechSegmenter==0.8.0 --no-deps

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

python -m spacy download es_core_news_md

python -m spacy download en_core_web_md

conda install --override-channels -c conda-forge ffmpeg

(torch se reinstala con CUDA 12.6 porque la versión por defecto no funciona en la GPU Tesla V100; ffmpeg sirve para recortar el audio y el vídeo que se descargan.)

Pasos para configurar las claves de las APIs:

cd equipo12

cp .env.example .env

Rellena en el .env las claves de The Guardian, YouTube y Podcast Index. Si falta alguna, esa fuente se omite y el resto funciona igual.

Pasos para probar la ingesta y los analizadores:

PYTHONPATH=. python -m pytest tests/test_ingesta.py tests/test_analyzers_unit.py tests/test_vocabulario.py -q

Si esto funciona, ejecuta ya todas las celdas de la libreta de jupyter my_app.ipynb. Antes de hacerlo:

- En la tercera celda elige AUDIT_DATA_MODE: "test_ui" (datos de prueba) o "real" (busca y analiza la empresa; tarda varios minutos).
- En la última celda cambia ids por el identificador de tu notebook (aparece en su URL).

Para seguir el progreso de la auditoría desde una terminal (en la carpeta equipo12):

tail -f logs/auditoria.log
