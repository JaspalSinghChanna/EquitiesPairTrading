FROM python:3.9
ENV DASH_DEBUG_MODE True
# Set the environmental variable in the image
ENV MONGO0=Gef2jyx6fC4fUF4b

COPY ./assets/ /assets/
COPY ./app.py /app.py
COPY ./requirements.txt /requirements.txt
COPY ./analytics_module.py /analytics_module.py
COPY ./data_loader.py /data_loader.py
COPY ./GoodTickers.csv /GoodTickers.csv
COPY ./Tickers.csv /Tickers.csv

WORKDIR /

RUN set -ex && \
    pip install -r requirements.txt
EXPOSE 8050
CMD ["python", "app.py"]