sudo -H pip3 install numpy scipy scikit-image theano lasagne
sudo -H pip3 install --upgrade https://github.com/Theano/Theano/archive/master.zip
sudo -H pip3 install --upgrade https://github.com/Lasagne/Lasagne/archive/master.zip

export THEANO_FLAGS='device=cuda,floatX=float32'

