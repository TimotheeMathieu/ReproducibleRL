launch_my_computation (){
    # Guix remove AVX2
    guix time-machine --url=https://codeberg.org/guix/guix.git --unsafe-channel-evaluation --channels=../channels.scm -- shell --with-graft=glibc=gcvi2 gcvi2 --max-jobs=1 --cores=1 -CNF python-pytorch bash python  python-numpy --share=$(realpath ..) -- python3 get_stats_torch.py  $1 "noavx"

    # Guix without removing AVX
    guix time-machine --url=https://codeberg.org/guix/guix.git --unsafe-channel-evaluation --channels=../channels.scm -- shell -CNF python-pytorch bash python  python-numpy bash --share=$(realpath ..) -- python3 get_stats_torch.py $1 "avx"
}

echo "Launch function created, you can now run launch_my_computation [output folder]."
