shopt -s nullglob
for experiment in "Acrobot" "Atari" "Mujoco"; do
    for env_type in "guix" "guix_with_masked_AVX" "pip_frozen" "pip_unfrozen" "conda_frozen" "conda_unfrozen"; do 
        COUNTER=0 
        machines=(~/ReproducibleRL/code/results/${experiment}/${experiment}_${env_type}/*)
        for machine in ${machines[@]} ; do
            if [ -d $machine ]; then
                if [ $(cat ${machine}/0.monitor.csv | wc -l) -gt 5 ]; then
                    COUNTER=$(expr $COUNTER + 1)
                fi
            fi
        done
        echo "${experiment} ${env_type} : ${COUNTER}"
    done
done
