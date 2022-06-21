FROM agrigorev/zoomcamp-model:mlops-3.9.7-slim

# Build image:
#   docker build -t mlops-zoomcamp-model:v1 .
# Run image:
#   docker run -it --name model mlops-zoomcamp-model:v1
#   docker run -d --name model mlops-zoomcamp-model:v1

RUN python -m pip install -U pip && \
    python -m pip install pipenv

WORKDIR /app
COPY . .

RUN pipenv install --system --deploy

ENTRYPOINT [ "python", "starter.py", "2021", "4"]