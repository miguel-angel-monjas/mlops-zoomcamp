FROM agrigorev/zoomcamp-model:mlops-3.9.7-slim

RUN python -m pip install -U pip && \
    python -m pip install pipenv

WORKDIR /app
COPY [ "Pipfile", "Pipfile.lock", "starter.py", "./" ]

RUN pipenv install --system --deploy

ENTRYPOINT [ "python", "starter.py", "2021", "4"]