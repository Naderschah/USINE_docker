# Start from root docker file
FROM rootproject/root:6.26.00-ubuntu20.04
# Install dependencies most will already be installed

#RUN apt-get update
#RUN apt-get install -y dpkg-dev
RUN apt-get update
RUN apt-get install -y git
#RUN apt-get install -y wget
#RUN apt-get install -y tar
#RUN apt-get install -y binutils
#RUN apt-get install -y gcc
#RUN apt-get install -y g++
#ENV DEBIAN_FRONTEND=noninteractive
#RUN apt-get -y install tzdata
#RUN apt-get -y install cmake
#RUN apt-get install -y binutils
#RUN apt-get install -y libx11-dev
#RUN apt-get install -y libxpm-dev
#RUN apt-get install -y libxft-dev
#RUN apt-get install -y libxext-dev
#RUN apt-get install -y python3
#RUN apt-get install -y libpng-dev
#RUN apt-get install -y libjpeg-dev
#RUN apt-get install -y libssl-dev
#RUN apt-get install -y gfortran
#RUN apt-get install -y libpcre3-dev
#RUN apt-get install -y xlibmesa-glu-dev
#RUN apt-get install -y libglew1.5-dev
#RUN apt-get install -y libftgl-dev
#RUN apt-get install -y libmariadb-dev
#RUN apt-get install -y libfftw3-dev
#RUN apt-get install -y libcfitsio-dev
#RUN apt-get install -y graphviz-dev 
#RUN apt-get install -y libavahi-compat-libdnssd-dev
#RUN apt-get install -y libldap2-dev
#RUN apt-get install -y libxml2-dev
#RUN apt-get install -y libkrb5-dev
#RUN apt-get install -y libgsl0-dev
#RUN apt-get install -y qtwebengine5-dev

# Installing mandatory packages CERN/ROOT already done with premade dockerfile
#RUN mkdir /ROOT
#RUN wget https://root.cern/download/root_v6.26.02.Linux-ubuntu20-x86_64-gcc9.4.tar.gz -O /ROOT/root.tar.gz
#RUN tar -xzvf /ROOT/root.tar.gz --directory /ROOT/
#SHELL ["/bin/bash", "-c"] 
#RUN source /ROOT/root/bin/thisroot.sh
#SHELL ["/bin/sh", "-c"] 


# Setting associated env variables
#ENV ROOTSYS=/ROOT/root/
ENV ROOTLIBS=$ROOTSYS/lib
ENV PATH=$PATH:$ROOTSYS/bin
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$ROOTSYS/lib
ENV MANPATH=$MANPATH:$ROOTSYS/man


# To clone from git
RUN git clone https://gitlab.com/dmaurin/USINE.git /USINE --branch V3.5
# Installation
#RUN mkdir USINE
WORKDIR /USINE
# mkdir build; cd build;   ../ 
RUN cmake -S /USINE/ -B /USINE/
# 1 is nr of cores
RUN make -j8 -C /USINE
# Set env variables
ENV USINE=/USINE
ENV PATH=$PATH:"$USINE/bin"
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$USINE/lib
ENV DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:$USINE/lib

# Set enviroment variable for AMS-02 data
ENV USINE_DATA=/data/USINE_FILES