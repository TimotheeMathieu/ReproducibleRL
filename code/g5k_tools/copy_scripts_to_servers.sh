#!/usr/bin/env bash
# Get array of cluster names

servers=(
    "grenoble"
    "lyon"
    "nancy"
    "rennes"
    "sophia"
)

print_help (){
printf "$0 |[username] : \n\ncopy repo to all servers on g5k\n\n"
}

if [[ $# -eq 0 ]] ; then
    print_help
    exit 0
fi

SERVER_USER_NAME=$1

GREEN='\033[1;32m'
NC='\033[0m'
read -r -p "This will remove all ReproducibleRL directory in g5k. Are you sure? [y/N] " response

case "$response" in
    [yY][eE][sS]|[yY]) 
        for server in "${servers[@]}"; do
            echo -e "${GREEN}Copying to ${server}${NC}"
            [[ ! -d /tmp/ReproducibleRL ]] && git clone --depth 1 https://gitlab.inria.fr/scool/ReproducibleRL /tmp/ReproducibleRL
            rm -r /tmp/ReproducibleRL/code/results
            rm -rf /tmp/ReproducibleRL/.git
            ssh $SERVER_USER_NAME@access.grid5000.fr rm -r /home/$SERVER_USER_NAME/${server}/ReproducibleRL || true
            scp -r /tmp/ReproducibleRL $SERVER_USER_NAME@access.grid5000.fr:/home/$SERVER_USER_NAME/${server}
        done
        ;;
        *)
          echo "Abort"
          ;;
esac
