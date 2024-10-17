#!/bin/bash
EV="./python/python38lungseg/"
APP="./sourcecode/src/vx/lungseg/"
SRC="./sourcecode/src/"
install () {
    #sudo apt-get install python3-venv
    #sudo apt install python3-pip
    #sudo apt-get install libsuitesparse-dev
    #sudo apt install libx11-dev
    #############sudo apt install nvidia-cuda-toolkit
    rm -r $EV
    mkdir $EV
    python3 -m venv $EV
    source $EV"bin/activate"
    #sudo apt-get install python3-tk
    pip3 install -r $APP"/zrequeriments.txt"
}
convert () {
    source $EV"bin/activate"
    cd $APP
    # python3 Convert.py
    python3 Convert2.py
}
lungseg () {
    source $EV"bin/activate"
    cd $APP
    python3 Lungseg.py
}
augm () {
    source $EV"bin/activate"
    cd $APP
    python3 augm.py
}
res () {
    source $EV"bin/activate"
    cd $APP
    python3 res.py
}


args=("$@")
T1=${args[0]}
FILEINPUT=${args[1]}
if [ "$T1" = "install" ]; then
    install
elif [ "$T1" = "convert" ]; then
    convert
elif [ "$T1" = "lungseg" ]; then
    lungseg
elif [ "$T1" = "augm" ]; then
    augm
elif [ "$T1" = "res" ]; then
    res
fi
