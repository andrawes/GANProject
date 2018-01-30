cd /scratch_net/biwidl103/aalbahou/exp0/improved_wgan_training
export SGE_GPU=`grep -h $(whoami) /tmp/lock-gpu*/info.txt | sed  's/^[^0-9]*//;s/[^0-9].*$//'`
python -u gan_mnist_caps.py > log.log
