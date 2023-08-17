FROM nvcr.io/partners/salesforce/warpdrive:v1.0

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get -y update && apt-get -y install ffmpeg gedit vim tmux net-tools apt-utils git htop
