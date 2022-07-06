# download large scale ood datasets
wget -P ${DATASETS} http://pages.cs.wisc.edu/~huangrui/imagenet_ood_dataset/iNaturalist.tar.gz
wget -P ${DATASETS} http://pages.cs.wisc.edu/~huangrui/imagenet_ood_dataset/SUN.tar.gz
wget -P ${DATASETS} http://pages.cs.wisc.edu/~huangrui/imagenet_ood_dataset/Places.tar.gz
wget -P ${DATASETS} https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz
# download small scale ood datasets
wget -P ${DATASETS} https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
wget -P ${DATASETS} https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz
wget -P ${DATASETS} https://www.dropbox.com/s/avgm2u562itwpkl/Imagenet.tar.gz
wget -P ${DATASETS} https://www.dropbox.com/s/kp3my3412u5k9rl/Imagenet_resize.tar.gz
wget -P ${DATASETS} https://www.dropbox.com/s/fhtsw1m3qxlwj6h/LSUN.tar.gz
wget -P ${DATASETS} https://www.dropbox.com/s/moqh2wh8696c3yl/LSUN_resize.tar.gz
wget -P ${DATASETS} https://www.dropbox.com/s/ssz7qxfqae0cca5/iSUN.tar.gz
wget -P ${DATASETS} https://people.eecs.berkeley.edu/~hendrycks/imagenet-a.tar
# svhn dataset
SVHN_DATASET_DIR="${DATASETS}/SVHN"
mkdir -P ${SVHN_DATASET_DIR}
wget -P ${SVHN_DATASET_DIR} http://ufldl.stanford.edu/housenumbers/test_32x32.mat
wget -P ${SVHN_DATASET_DIR} http://ufldl.stanford.edu/housenumbers/train_32x32.mat
wget -P ${SVHN_DATASET_DIR} http://ufldl.stanford.edu/housenumbers/extra_32x32.mat
# unpack
tar -xf ${DATASETS}/cifar-10-python.tar.gz -C ${DATASETS}
tar -xf ${DATASETS}/cifar-100-python.tar.gz -C ${DATASETS}
tar -xf ${DATASETS}/Imagenet.tar.gz -C ${DATASETS}
tar -xf ${DATASETS}/Imagenet_resize.tar.gz -C ${DATASETS}
tar -xf ${DATASETS}/LSUN.tar.gz -C ${DATASETS}
tar -xf ${DATASETS}/LSUN_resize.tar.gz -C ${DATASETS}
tar -xf ${DATASETS}/iSUN.tar.gz -C ${DATASETS}
tar -xf ${DATASETS}/imagenet-a.tar -C ${DATASETS}
tar -xf ${DATASETS}/iNaturalist.tar.gz -C ${DATASETS}
tar -xf ${DATASETS}/SUN.tar.gz -C ${DATASETS}
tar -xf ${DATASETS}/Places.tar.gz -C ${DATASETS}
tar -xf ${DATASETS}/dtd-r1.0.1.tar.gz -C ${DATASETS}
# delete archives
rm ${DATASETS}/*.tar ${DATASETS}/*.tar.gz 

# download checkpoints
wget -P ${MODELS} https://www.dropbox.com/s/mx9gytxj39241on/checkpoints.zip
unzip ${MODELS}/checkpoints.zip -d ${MODELS}
rm ${MODELS}/checkpoints.zip
