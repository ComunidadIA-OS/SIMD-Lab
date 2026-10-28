Pasos para crear el entorno virtual (Python 3.11, 3.12 o 3.13):
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
