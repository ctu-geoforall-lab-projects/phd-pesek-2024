FROM tensorflow/tensorflow:2.4.0rc2

ARG DEBIAN_FRONTEND=noninteractive

RUN apt update

RUN mkdir /tmp/src

# install pip dependecies
COPY requirements.txt /tmp/src/requirements.txt
RUN pip install -r /tmp/src/requirements.txt

# git clone JEODPP packages
RUN apt install git -y

RUN cd /tmp/src && \
    git clone https://gitlab+deploy-token-285067:ygASxr_ATUQTT85YJTHN@gitlab.com/ctu-geoforall-lab/jeodpp-packages.git

# install mialib dependencies
RUN apt install -yq \
    build-essential \
    libgsl-dev \
    libgsl0-dev \
    libfann-dev \
    libgeotiff-dev \
    libfftw3-dev \
    libproj-dev \
    libjsoncpp-dev \
    libgdal-dev \
    libssl-dev \
    doxygen \
    swig \
    uthash-dev \
    texlive \
    libshp-dev \
    texlive-latex-extra

# install mialib
RUN cd /tmp/src/jeodpp-packages/mia && \
    make && \
    make install

# install jiplib dependencies
RUN apt install cmake libboost-filesystem-dev libboost-serialization-dev python3-numpy python3-gdal -y

# install jiplib
RUN cd /tmp/src/jeodpp-packages/jiplib && \
    mkdir build && \
    cd build && \
    cmake .. -DCMAKE_MODULE_PATH=/tmp/src/jeodpp-packages/jiplib/ && \
    make -j && \
    make install

# install pyJEO
RUN cd /tmp/src/jeodpp-packages/pyjeo && \
    python setup.py install

# clean the mess
RUN rm -r /tmp/src

CMD ["bash"]
