FROM tensorflow/tensorflow:2.4.1

ARG DEBIAN_FRONTEND=noninteractive

ENV CPU_NR 8
ENV SQLITE_VERSION 3340000
ENV SQLITE_YEAR 2020
ENV CURL_VERSION 7.75.0
ENV LIBTIFF_VERSION 4.1.0
ENV LIBGEOTIFF_VERSION 1.6.0
ENV XERCES_VERSION 3.2.3
ENV GEOS_VERSION 3.9.1
ENV PROJ_VERSION 7.2.0
ENV GDAL_VERSION 3.2.1

RUN apt update

RUN mkdir /tmp/src

# install basic stuff
RUN apt install wget pkg-config bash-completion -y
RUN apt install libz-dev openssl libtool m4 automake doxygen -y

# install pip dependecies
COPY requirements.txt /tmp/src/requirements.txt
RUN pip install -r /tmp/src/requirements.txt

# install sqlite
RUN cd /tmp/src && \
    wget https://sqlite.org/${SQLITE_YEAR}/sqlite-autoconf-${SQLITE_VERSION}.tar.gz && \
    tar -xzf sqlite-autoconf-${SQLITE_VERSION}.tar.gz && \
    cd sqlite-autoconf-${SQLITE_VERSION} && \
    ./configure --prefix=/usr/local && \
    make -j${CPU_NR} && \
    make install

# install curl
RUN cd /tmp/src && \
    wget https://curl.haxx.se/download/curl-${CURL_VERSION}.tar.gz && \
    tar -xzf curl-${CURL_VERSION}.tar.gz && \
    cd curl-${CURL_VERSION} && \
    ./configure --prefix=/usr/local && \
    make -j${CPU_NR} && \
    make install

# install libtiff
RUN cd /tmp/src && \
    wget https://download.osgeo.org/libtiff/tiff-${LIBTIFF_VERSION}.tar.gz && \
    tar -xzf tiff-${LIBTIFF_VERSION}.tar.gz && \
    cd tiff-${LIBTIFF_VERSION} && \
    ./configure --prefix=/usr/local && \
    make -j${CPU_NR} && \
    make install

# install PROJ
RUN cd /tmp/src && \
    wget https://download.osgeo.org/proj/proj-${PROJ_VERSION}.tar.gz && \
    tar -zvxf proj-${PROJ_VERSION}.tar.gz && \
    cd proj-${PROJ_VERSION} && \
    ./configure --disable-dependency-tracking && \
    make -j${CPU_NR} && \
    make install

# install xerces
RUN cd /tmp/src && \
    wget https://github.com/apache/xerces-c/archive/v${XERCES_VERSION}.tar.gz && \
    tar -zxvf v${XERCES_VERSION}.tar.gz && \
    cd xerces-c-${XERCES_VERSION} && \
    ./reconf && \
    ./configure && \
    make -j${CPU_NR} && \
    make install

# install geotiff
RUN cd /tmp/src && \
    wget -q https://github.com/OSGeo/libgeotiff/releases/download/${LIBGEOTIFF_VERSION}/libgeotiff-${LIBGEOTIFF_VERSION}.tar.gz && \
    tar -xzf libgeotiff-${LIBGEOTIFF_VERSION}.tar.gz && \
    cd libgeotiff-${LIBGEOTIFF_VERSION} && \
    ./configure --prefix=/usr/local && \
    make -j${CPU_NR} && \
    make install

# install geos
RUN cd /tmp/src && \
    wget http://download.osgeo.org/geos/geos-${GEOS_VERSION}.tar.bz2 && \
    bunzip2 geos-${GEOS_VERSION}.tar.bz2 && \
    tar xvf  geos-${GEOS_VERSION}.tar && \
    cd geos-${GEOS_VERSION} && \
    ./configure && \
    make -j${CPU_NR} && \
    make install

# install GDAL
RUN cd /tmp/src && \
    wget http://download.osgeo.org/gdal/${GDAL_VERSION}/gdal-${GDAL_VERSION}.tar.gz && \
    tar -zvxf gdal-${GDAL_VERSION}.tar.gz && \
    cd gdal-${GDAL_VERSION} && \
    PKG_CONFIG_PATH=/usr/lib/pkgconfig:/usr/local/lib/pkgconfig \
    ./configure --with-threads --with-libtiff=/usr/local \
    --with-geotiff=/usr/local --with-jpeg=internal \
    --with-gif=internal --with-png=internal --with-libz=/usr --with-python \
    --with-openjpeg --with-xerces --with-geos && \
    make -j${CPU_NR} && \
    make install && \
    ldconfig

RUN pip install gdal==${GDAL_VERSION}

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
#    libgeotiff-dev \
    libfftw3-dev \
#    libproj-dev \
    libjsoncpp-dev \
#    libgdal-dev \
    libssl-dev \
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
RUN apt install cmake libboost-filesystem-dev libboost-serialization-dev python3-numpy -y

# install jiplib
RUN cd /tmp/src/jeodpp-packages/jiplib && \
    mkdir build && \
    cd build && \
    cmake .. -DCMAKE_MODULE_PATH=/tmp/src/jeodpp-packages/jiplib/ && \
    make -j${CPU_NR} && \
    make install

# install pyJEO
RUN cd /tmp/src/jeodpp-packages/pyjeo && \
    python setup.py install

# clean the mess
RUN rm -r /tmp/src

CMD ["bash"]
