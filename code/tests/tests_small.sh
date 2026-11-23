#!/bin/bash 

# Get some colors :)
GREEN='\033[1;32m'
RED='\033[0;31m'
NC='\033[0m'

export PYENV_ROOT="/tmp/pyenv"
[[ -d $PYENV_ROOT ]] && rm -r $PYENV_ROOT --force
curl -fsSL https://pyenv.run | bash
export PATH="/tmp/pyenv/bin:$PATH"
eval "$(pyenv init - bash)"
pyenv install -s 3.12.12

export PATH="/tmp/pyenv/versions/3.12.12/bin:$PATH"
# Utility functions

raise_and_conclude (){
status=$?
if [ $status -eq 0  ]; then
    echo -e "${GREEN} $1 passed ! ${NC}"
else
    echo -e "${RED} $1 failed :( ! ${NC}"
    # exit 1
fi
deactivate
rm -r /tmp/pip-unpack*
}

make_venv (){
REQUIREMENT_FILE=$1
[[ -d /tmp/venv_reproductibility_test ]] && rm --force -r  /tmp/venv_reproductibility_test
python3 -m venv /tmp/venv_reproductibility_test
source /tmp/venv_reproductibility_test/bin/activate
python3 -m pip install -r $REQUIREMENT_FILE &>>  /tmp/test.logs
}

# Do the tests
echo "Test logs can be found in  /tmp/test.logs"
[[ -f /tmp/test.logs ]] && rm /tmp/test.logs

for experiment in "Mujoco" ; do
    echo -e "${GREEN}Launching experiment ${experiment}${NC}"
    mkdir -p /tmp/results_reprorl/$experiment
    cd ../${experiment}/${experiment}_pip_frozen
    make_venv requirements.txt
    python3 ../run_ppo.py /tmp/results_reprorl/$experiment --n-timesteps 500 &>> /tmp/test.logs
    raise_and_conclude "${experiment}"
    cd ../../tests
    
done

echo -e "${GREEN}Launching experiment Acrobot_GPU${NC}"
cd ../Acrobot_GPU/ 
make_venv requirements.txt 
CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 run_ppo.py /tmp/results_reprorl/Acrobot_GPU --n-timesteps 500  &>>  /tmp/test.logs
raise_and_conclude "Acrobot_GPU"
cd ../tests

echo -e "${GREEN}Launching experiment fitted_Q${NC}"
cd ../fitted_q/ 
make_venv requirements.txt
python3 run_fitted_q.py /tmp/results_reprorl/Fitted_Q --n-timesteps 500   &>>  /tmp/test.logs
raise_and_conclude "fitted_Q"


echo -e "${GREEN}Launching experiment fitted_Q_torch${NC}"
make_venv requirements.txt 
python3 run_fitted_q_torch.py /tmp/results_reprorl/Fitted_Q --n-timesteps 500    &>>  /tmp/test.logs
raise_and_conclude "fitted_Q_torch"
cd ../tests

echo -e "${GREEN}Launching experiment cost_repro${NC}"
cd ../cost_reproducibility/ 
make_venv requirements.txt 
python3 run_ppo.py "pip" /tmp/results_reprorl/cost_repro --n-timesteps 500   &>>  /tmp/test.logs
raise_and_conclude "cost_repro"
cd ../tests
 
rm -r /tmp/venv_reproductibility_test
