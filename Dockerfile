FROM nvcr.io/partners/salesforce/warpdrive:v1.0

RUN apt-get update && apt-get -y --no-install-recommends install ffmpeg