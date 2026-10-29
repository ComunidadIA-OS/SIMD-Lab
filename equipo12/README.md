
# Documentación de Arquitectura: Equipo SIMD Lab

Trabajo desarrollado por Celia Almendros Saelices, Esther Almendros Saelices y Teresa Valero Díaz.

## 1. Propósito de la Plataforma, Métricas y Fórmulas de Ponderación

### Objetivo del Sistema

La plataforma está diseñada para auditar la presencia, relevancia y tratamiento de género en los contenidos multimedia corporativos de empresas e instituciones públicas o privadas. Evalúa en qué medida se da voz, visibilidad y liderazgo a las mujeres a través de cuatro formatos complementarios: **texto**, **audio**, **imagen** y **vídeo**, asegurando un análisis alineado con criterios de sostenibilidad social y gobernanza (ESG) y el reglamento europeo de inteligencia artificial (**EU AI Act**).

---

### Métricas Analizadas por Modalidad

1. **Texto (9 métricas):**
* *Proporción de frases*: porcentaje de frases referidas a mujeres (`sentence_female_ratio`), hombres (`sentence_male_ratio`) y neutras (`sentence_neutral_ratio`).


* *Diversidad léxica y sufijos*: variedad de raíces léxicas femeninas (`lexical_root_diversity`) y sufijos que marcan género (`feminine_suffix_ratio`).


* *Frecuencia de mención*: recuento absoluto normalizado frente al total del texto (`relative_frequency_score` y `female_mention_frequency`).


* *Agencia sintáctica*: mide si la mujer aparece como sujeto activo de las acciones verbales (`syntactic_agency_ratio`).


* *Voz experta y titular*: frecuencia con la que se citan declaraciones directas como especialista (`expert_quote_ratio`) y presencia en titulares (`headline_presence`).




2. **Audio (7 métricas):**
* *Apertura y cierre*: si una voz femenina abre (`female_opened_score`) o despide (`female_closed_score`) la locución o programa.


* *Tiempo y turnos de palabra*: porcentaje de tiempo de voz femenina (`female_speaking_time_ratio`) y proporción de turnos tomados (`speaking_turns_ratio`).


* *Presencia en el tramo central*: participación en el 20%-80% central de la grabación (`middle_segment_female_ratio`).


* *Fiabilidad acústica y no interrupción*: certeza del modelo sobre el género de la voz (`acoustic_gender_confidence`) y fluidez sin ser pisada o interrumpida (`no_interruption_score`).




3. **Imagen (8 métricas):**
* *Presencia visual*: ratio de rostros y cuerpos femeninos (`visual_gender_female_ratio`).


* *Composición y lenguaje visual*: escala del plano (`shot_scale_importance_score`), ángulo de cámara que denote autoridad (`camera_angle_power_score`), ubicación según la regla de los tercios (`rule_of_thirds_alignment`) e iluminación/sombras (`shadow_lighting_prominence`).


* *Centralidad y contexto*: posición centrada en la composición (`spatial_centrality_score`), persistencia entre fotos de la misma galería (`cross_image_persistence_ratio`) y contexto visual en puestos de liderazgo (`leadership_visual_context`).




4. **Vídeo (5 métricas):**
* *Pista de audio y fotogramas*: hereda y evalúa tanto la pista de audio como las capturas visuales de fotogramas clave.


* *Tiempo en pantalla*: porcentaje del tiempo total que aparece una mujer en cámara (`female_screen_time_ratio`).


* *Dinamismo y sincronía*: movimiento mediante flujo óptico (`optical_flow_motion_score`), dominancia cinética (`kinetic_dominance_ratio`), cambios en silueta (`silhouette_change_histogram_score`) y sincronización entre quien habla y la imagen activa (`active_speaker_sync_score`).





---

### Fórmulas de Cálculo y Ponderación

El sistema asigna una puntuación entre **0 y 100** a cada modalidad. Si en alguna métrica no hay datos (por ejemplo, si nadie interrumpe en un audio o no hay declaraciones directas en un texto), **no se penaliza asignando un cero ni un valor ficticio**, sino que ese indicador se descarta y su peso se redistribuye estrictamente entre los que sí tienen medición real:

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

Para cada documento individual, se calcula la media aritmética simple de las modalidades que efectivamente contiene. Si un documento no posee ninguna modalidad analizable, su valor queda registrado como no disponible (`NaN`), evitando inventar predicciones arbitrarias:

$$\text{Puntuación Documento} = \begin{cases} \text{Media}\left(\{\text{Score}_m \mid m \in \text{Modalidades Presentes}\}\right) & \text{si hay modalidades presentes} \\  \text{NaN} & \text{si no hay datos directos} \end{cases}$$

#### Índice Global Ponderado de la Entidad:

El **Índice Global Ponderado** (el indicador principal de la entidad) consolida las cuatro modalidades ponderando cada una según su **volumen de presencia efectiva** (el número de documentos válidos de ese canal, $N_m$):

$$\text{Índice Global Ponderado} = \frac{\sum_{m \in \{\text{Texto, Audio, Imagen, Vídeo}\}} N_m \cdot \text{Score}_m}{\sum_{m} N_m}$$



De este modo, si la auditoría ha analizado mayoritariamente artículos de prensa, el canal de texto tendrá un peso proporcionalmente mayor en la nota global, coincidiendo con los porcentajes de peso exhibidos en el panel principal.

---

## 2. Fuentes de Datos y Conexión con Servicios Externos

La plataforma recopila información a través de clientes modulares en `src/connectors/`:

1. **The Guardian (`guardian_client.py`):** Conecta con la Open Platform de The Guardian para obtener noticias, artículos de opinión, autores, textos íntegros y enlaces a fotografías de prensa.


2. **YouTube (`youtube_client.py`):** Extrae metadatos de canales oficiales, fragmentos de vídeo de ruedas de prensa y debates corporativos mediante la API oficial de YouTube y utilidades de extracción.


3. **Wikimedia Commons (`wikimedia_client.py`):** Recupera fotografías de directivos, consejos de administración y eventos corporativos con licencias abiertas y metadatos estructurados.


4. **Internet Archive (`internet_archive_client.py`):** Consulta grabaciones sonoras históricas, vídeos de archivo y hemeroteca digital.


5. **Podcast Index (`podcastindex_client.py`):** Busca entrevistas y episodios de podcast asociados a la organización analizada.


6. **Flickr (`flickr_client.py`):** Permite complementar la búsqueda de imágenes bajo licencias Creative Commons.



---

## 3. Extracción de Métricas, Modelos y Procesamiento Previo

Antes de extraer métricas definitivas, los datos pasan por varias etapas de acondicionamiento:

### Procesamiento Previo y Protección de Privacidad

* **Preservación y Minimización de Datos (`privacy_scrubber.py`):** Las transcripciones y textos se limpian de datos de carácter personal (PII) mediante reglas de anonimización (sustituyendo identificadores por etiquetas anónimas).


* **Ciclo de Vida de Multimedia Efímera:** Los archivos multimedia descargados se almacenan temporalmente en `/tmp/ramdisk/` exclusivamente durante la sesión de auditoría para permitir su verificación auditada y explicabilidad visual (LIME/reproducción local). Se purgan automáticamente al iniciar una nueva búsqueda o voluntariamente mediante el botón de cierre en el panel de gobernanza.


* **Recorte multimedia con FFmpeg:** El audio se normaliza a formato mono de 16 kHz para los modelos de voz, y los vídeos se segmentan extrayendo fotogramas representativos a 1 imagen por segundo (1 fps).



### Modelos de Inteligencia Artificial Utilizados

* **Texto (`TextGenderAnalyzer`):**
* Emplea modelos lingüísticos de **spaCy** (`es_core_news_md` y `en_core_web_md`) para el análisis sintáctico de dependencias (para determinar quién es el sujeto activo de la acción), etiquetado morfológico y concordancia léxica apoyado por un vocabulario de género específico (`vocabulario_genero.py`).




* **Audio (`AudioGenderAnalyzer`):**
* Utiliza **inaSpeechSegmenter**, una red neuronal convolucional entrenada para segmentar señales acústicas en tramos de música, silencio y habla diferenciada entre hombre y mujer.




* **Imagen y Vídeo (`VisionGenderAnalyzer`):**
* Utiliza **YuNet** (detección facial ligera y de alta velocidad) para localizar rostros en imágenes y fotogramas.


* Utiliza **OpenCLIP** (redes multimodales de contraste visión-lenguaje) para clasificar atributos de contexto, puestos de responsabilidad y composición de la imagen (ángulos, planos y entorno).





---

## 4. Estructura del Modelo de Datos (Schemas)

El proyecto utiliza **Pydantic** (`src/common/schemas.py`) para validar estrictamente la información entre fases:

* **`UnifiedMediaDocument`:** Representa un documento tras la descarga de la fuente. Contiene:


* Identificador único (`doc_id`), fuente (`source`), término buscado (`company_query`), URL y título.


* Texto en bruto (`raw_text`), fecha de publicación, autor e idioma normalizado en formato ISO 639-1.


* Listas de enlaces remotos (`image_urls`, `audio_urls`, `video_urls`) y rutas locales a los archivos temporales (`image_paths`, `audio_paths`, `video_paths`, `frame_dirs`).


* Modalidades disponibles (`available_modalities`).




* **Métricas Especializadas:** Clases `TextMetrics`, `AudioMetrics`, `ImageMetrics` y `VideoMetrics` que agrupan los valores numéricos calculados (todos normalizados entre 0.0 y 1.0).


* **`ExtractedFeaturesPayload`:** Objeto central consolidado. Reúne las métricas de las 4 modalidades, conserva el texto completo (`full_raw_text`) y las muestras locales (`media_image_samples`, `media_audio_samples`, etc.), facilitando su exportación tabular y su posterior inspección en el módulo de explicabilidad.



---

## 5. Pantallas de la Interfaz Gráfica (Streamlit)

La aplicación web (`frontend/app.py`) ofrece un entorno visual estructurado en una barra lateral y tres pestañas temáticas:

1. **Barra Lateral (`search_sidebar.py`):**
* Permite introducir el nombre de la empresa a auditar, activar o desactivar fuentes de datos individuales, configurar el límite de documentos y alternar entre modo real o modo de prueba rápida.




2. **Pestaña 1: Panel ESG (`tab_esg_dashboard.py`):**
* Muestra el Índice Global Ponderado de la entidad auditada, los pesos relativos por presencia, desgloses por canal, tarjetas con indicadores clave de paridad y gráficos comparativos de participación.




3. **Pestaña 2: Explicabilidad con SHAP y LIME (`tab_xai_shap_lime.py`):**
* Presenta gráficos de impacto para entender por qué la puntuación subió o bajó. Permite alternar entre el resumen del artículo completo o frases específicas, inspeccionar superpíxeles con LIME en imágenes reales y verificar fragmentos sonoros y clips de vídeo.




4. **Pestaña 3: Cumplimiento Normativo y Equidad (`tab_ai_act_fairness.py`):**
* Resume el nivel de riesgo de acuerdo al marco del EU AI Act, ofrece un editor interactivo para supervisión humana (*Human-in-the-Loop*), desglosa el cómputo y consumo de $\text{CO}_2$, y permite descargar el certificado formal de auditoría en formato JSON.





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
                  │      Descarga los archivos multimedia de los documentos seleccionados en /tmp/ramdisk.
                  │
                  ├─► 4. multimodal_pipeline.py
                  │      ├─► privacy_scrubber.py (anonimiza datos personales)
                  │      ├─► text_analyzer.py    (extrae métricas de texto con spaCy)
                  │      ├─► audio_analyzer.py   (segmenta voces con inaSpeechSegmenter)
                  │      └─► vision_analyzer.py  (evalúa planos y rostros con YuNet y OpenCLIP)
                  │
                  ├─► 5. scoring_model.py
                  │      Calcula las medias por modalidad, los scores por documento y 
                  │      el Índice Global Ponderado exacto según la presencia real.
                  │
                  ├─► 6. shap_explainer.py / lime_explainer.py
                  │      Calcula analíticamente los valores Shapley y los superpíxeles LIME.
                  │
                  ├─► 7. audit_logger.py
                  │      Registra la huella de ejecución y el gasto energético en tiempo real.
                  │
                  └─► 8. Retorno a app.py
                         Renderiza los resultados, gráficos y alertas en pantalla.

```

---

## 7. Explicabilidad, Supervisión Humana y Cumplimiento Normativo

### Explicabilidad (XAI)

* **SHAP Analítico Exacto (`shap_explainer.py`):**
Al ser el modelo de puntuación una función lineal aditiva ponderada, SHAP calcula directamente los valores de Shapley analíticos exactos:

$$\phi_i = w_i \cdot (x_i - \text{base}_i)$$



donde $\text{base}_i = 0.5$ representa el punto neutro de paridad (50%). Si un documento carece de datos en una métrica, su impacto asignado es exactamente cero, eliminando ruidos, muestreos aleatorios o entrenamientos sintéticos.
* **LIME Imagen (`lime_explainer.py`):**
Genera explicaciones visuales locales segmentando la fotografía real en superpíxeles y delimitando mediante contornos las regiones anatómicas o espaciales decisivas que determinaron la composición.
* **Evidencia Textual (`text_evidence.py`):**
Localiza y resalta en el texto íntegro las menciones, cargos y pronombres detectados, indicando si actúan como sujeto activo de la acción o como citas de opinión experta.

### Supervisión Humana (*Human-in-the-Loop*)

* A través de `human_oversight.py` y la tabla editable de la interfaz, la plataforma no actúa como un sistema cerrado. Un auditor experto puede revisar las puntuaciones automáticas, ajustar manualmente los pesos o las notas medias de cada canal y justificar discrepancias.


* Al aplicar cambios manuales, el sistema recalcula el Índice Global Ponderado, registra una traza inmutable de auditoría con el número de filas modificadas y sella la fecha y hora UTC de la validación.



### Cumplimiento del EU AI Act y RGPD

* **Trazabilidad y Registro de Auditoría (Art. 11 y 12 EU AI Act):** Toda decisión, parámetro de entrada, puntuación ajustada y resultado queda estructurado en un Certificado Oficial en formato JSON listo para su descarga y verificación regulatoria.


* **Transparencia Algorítmica (Art. 13 EU AI Act):** Ausencia de cajas negras u opacidad en el scoring; las fórmulas matemáticas y desgloses de SHAP y LIME son 100% auditables.


* **Supervisión Humana (Art. 14 EU AI Act):** Capacidad operativa de intervención directa y recálculo del baremo por parte de un operador humano.


* **Minimización de Datos y Privacidad por Diseño (Art. 5.1.c y Art. 25 RGPD):** Aislamiento de muestras en almacenamiento temporal efímero con opción de purga manual o automática tras concluir la sesión, evitando la acumulación de datos biométricos crudos en disco.


* **Sostenibilidad y Huella Ambiental (Anexo IV EU AI Act):** Utiliza la librería **CodeCarbon** y telemetría de hardware para medir en tiempo real el consumo de CPU, memoria y estimar los gramos de $\text{CO}_2$ equivalentes emitidos durante la inferencia.


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
