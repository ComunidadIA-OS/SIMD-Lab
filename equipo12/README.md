Pasos para crear el entorno virtual (Python 3.11, 3.12 o 3.13):
Abre una terminal en la carpeta raíz del repositorio (SIMD-Lab) y ejecuta los siguientes comandos en orden:

pip install --user virtualenv

python3 -m virtualenv .venv

source .venv/bin/activate

pip install -r equipo12/requirements/requirements.txt

pip install inaSpeechSegmenter==0.8.0 --no-deps 

python -m spacy download es_core_news_md

python -m spacy download en_core_web_md

Pasos para probar el generador Mock, LightGBM y Fairlearn y construir la imagen docker:

cd equipo12
PYTHONPATH=. python tests/test_xai_fairness_mock.py


Si esto funciona, ejecuta ya todas las celdas de la libreta de jupyter my_app.ipynb.
