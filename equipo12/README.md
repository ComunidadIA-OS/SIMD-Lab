Pasos para crear el entorno virtual:
Abre una terminal y ejecuta los siguientes comandos en orden:

pip install --user virtualenv

python3 -m virtualenv .venv

source .venv/bin/activate

pip install -r equipo12/requirements/requirements.txt

Pasos para probar el generador Mock, LightGBM y Fairlearn y construir la imagen docker:

cd equipo12
PYTHONPATH=. python equipo12/tests/test_xai_fairness_mock.py


Si esto funciona, ejecuta ya todas las celdas de la libreta de jupyter my_app.ipynb.
