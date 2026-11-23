
mkdir -p ../results/xp_atari_local

cd ../Atari/Atari_pip_frozen && source setup_environment.sh

for run_number in $(seq 1 10); do
    launch_my_computation ../../results/xp_atari_local/xp_atari_local_repro_${run_number}/
done


##### compare results
PARENT="../../results/xp_atari_local"
reference=$(echo "$PARENT"/xp_atari_local_repro_1/**/0.monitor.csv)

extract_r_l_column() {
    awk -F',' '
        !/^#/ && $1 != "r" {
            print $1 "," $2
        }
    ' "$1"
}

reference_data=$(extract_r_l_column "$reference")

identical=true

for file in "$PARENT"/**/0.monitor.csv; do
    [ -f "$file" ] || continue

    if [ "$(extract_r_l_column "$file")" != "$reference_data" ]; then
        echo "❌ Différent : $file"
        identical=false
        echo $(extract_r_l_column "$file")
        echo $reference_data
    else
        echo "✅ Identique : $file"
    fi
done

if $identical; then
    echo "✅ Tous les résultats r,l sont identiques."
else
    echo "❌ Les résultats r,l diffèrent."
fi


##### end compare results

cd -
echo "Results in ../results/xp_atari_local_repro/"
