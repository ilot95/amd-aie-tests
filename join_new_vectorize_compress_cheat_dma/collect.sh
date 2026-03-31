for elements in 1024 2048 4096 8192 16384 32768 65536 131072 262144
do
    make clean && make run_xchesscc hostElements=${elements}
done
