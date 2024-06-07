#!/bin/bash
read -p "Input dataset :(default valule: github[gcj | github]) " dataset
dataset="${dataset:='github'}"
if [[ ${dataset,,} == 'github' ]]; then
    read -p "Input dataset folder root dir:(default valule: ../datasets/repos_24) " dbroot
    dbroot="${dbroot:='../datasets/repos_24'}"
    read -p "Input a name of a folder in 'data' folder: (default valule: github_repos_24_15authors_80_700_weighted)" dataName
    dataName="${dataName:='github_repos_24_15authors_80_700_weighted'}"
    # read -p "Input the start step (default valule: 1) [1 .. 5]" start
    # start="${start:=1}"
    read -p "Input the number of balance (default value: 0)" numOfBalance
    numOfBalance="${numOfBalance:=0}"
elif [[ ${dataset,,} == 'gcj' ]]; then
    root='/media/yihua/2TB/backup'
    read -p "Input folder name in 'data':(default valule: 50_exact_opt0) " dbname
    dbname="${dbname:=50_exact_opt0}"
    # dbname=$1
    read -p "Input number of authors:(default valule: 50) " numAuthors
    numAuthors="${numAuthors:=50}"
    # read -p "Input the start step (default valule: 1) [1 .. 5]" start
    # start="${start:=1}"
    dbroot="$root/$((numAuthors))Authors"
    read -p "Input the attack flag default valule: 0) [0 | 1]" flag
    flag="${flag:=0}"
else
    echo "illegal input"
    exit 1
fi

read -p "Input the start step (default valule: 1) [1 .. 5]" start
start="${start:=1}"
read -p "Input the number of threads (default value: 8) [1 .. 8]" numOfThreads
numOfThreads="${numOfThreads:=8}"

# dbroot="$root/$dbname"
cwd=$(pwd)

function getAllSlices {
    # after successfully running the script, we create a file to set a flag
    gnome-terminal -e "bash -c -i 'workon angr-dev; python3 getAllBinSlices.py $1 && sleep 10 && echo getAllSlices\ finished > /tmp/getAllSlices_finished$2'"
    # gnome-terminal -e "bash -c -i 'workon -h; workon angr-dev; python3 getAllBinSlices.py -h && sleep 10 && echo getAllSlices\ finished > /tmp/getAllSlices_finished$2'"
}

function decompileAllSlices {
    # after successfully running the script, we create a file to set a flag
    gnome-terminal -e "bash -c -i 'workon angr-dev; python3 decompileAllSlices.py $1 && sleep 10  && echo decompile\ done > /tmp/decompileAllSlices_finished$2'"
    # gnome-terminal -e "bash -c -i 'workon -h; workon angr-dev; python3 decompileAllSlices.py $1 && sleep 10  && echo decompile\ done > /tmp/decompileAllSlices_finished$2'"
}

function getDotFiles {
    # after successfully running the script, we create a file to set a flag
   gnome-terminal -e "bash -c -i 'conda activate joern; python3 joern_processor_combin.py $1; sleep 10  && echo joern\ done > /tmp/getDotFiles_finished$2'"
    # gnome-terminal -e "bash -c -i 'conda -h; conda activate joern; python3 joern_processor_combin.py $1; sleep 30  && echo joern\ done > /tmp/getDotFiles_finished$2'"
}

function getASTnPDG {
    # after successfully running the script, we create a file to set a flag
    gnome-terminal -e "bash -c -i 'conda activate diffpool; python getASTnPDG.py $1 ; sleep 10  && echo get\ all\ dots > /tmp/getASTnPDG_finished$2'"
    # gnome-terminal -e "bash -c -i 'conda -h; conda activate diffpool; python getASTnPDG.py $1 ; sleep 30  && echo get\ all\ dots > /tmp/getASTnPDG_finished$2'"
}

function removeErrorDots {
    # after successfully running the script, we create a file to set a flag
    gnome-terminal -e "bash -c -i 'conda activate diffpool; python auto_run_getASTorPDG.py $1 ; sleep 10  && echo remove\ wrong\ dots > /tmp/removeErrorDots_finished$2'"
    # gnome-terminal -e "bash -c -i 'conda activate diffpool; python auto_run_getASTorPDG.py -h && sleep 10 && echo remove\ wrong\ dots > /tmp/removeErrorDots_finished$2'"
}

function waitFlagFiles {
    for ((i=1; i<=$1; i++)); do
        while [[ ! -f "/tmp/$2_finished$i" ]]; do
            sleep 1
        done
    done
}

# clean all flag files at the beginning of the script
find /tmp -maxdepth 1 -type f -iname '*_finished?' -exec rm {} \;
start_time=$(date +%s)

# to slice binary files
step1=1
if [[ ${dataset,,} == 'github' ]]; then
    #statements
    binDir="$dbroot/repos"
    logs="$dbroot/slicesLogs"
else
    binDir="$dbroot/$dbname/AuthorsDirectory/"
    # binDir="$dbroot/AuthorsDirectory/"
    logs="$dbroot/$dbname/slicesLogs"
    # logs="$dbroot/slicesLogs"
fi

if [[ $start -le $step1 ]]; then
    for i in $(seq 0 $((numOfThreads-1))); do
        args="-d $binDir -l $logs -n $numOfThreads -i $i"
        echo "getAllSlices $args $((i+1)) &"
        getAllSlices "$args" "$((i+1))" &
    done

    waitFlagFiles "$numOfThreads" "getAllSlices"

    # remove flag files
    find /tmp -maxdepth 1 -type f -iname 'getAllSlices_finished?' -exec rm {} \;
    echo 'finished getAllSlices' 
fi

# to decompile all sliced files
step2=$((step1 + 1))

slicesDir=$binDir
# errLogDir="$dbroot/$dbname/decompilLogs"
if [[ ${dataset,,} == 'github' ]]; then
    #statements
    errLogDir="decompilLogs"
else
    errLogDir="$dbroot/$dbname/decompilLogs"
fi

if [[ $start -le $step2 ]]; then
    for i in $(seq 0 $((numOfThreads-1))); do
        args="-d $slicesDir -e $errLogDir -n $numOfThreads -i $i"
        echo "decompileAllSlices $args $((i+1)) &"
        decompileAllSlices "$args" "$((i+1))" &
    done

    # wait
    waitFlagFiles "$numOfThreads" "decompileAllSlices"

    # find all finished files and delete them; '\;' indicates the command ends.
    find /tmp -maxdepth 1 -type f -iname 'decompileAllSlices_finished?' -exec rm {} \;

    echo "decompileAllSlices done"
fi

# to use joern to get dot files
step3=$((step2 + 1))
pesudoDir=$slicesDir
# dotDir="$dbroot/$dbname/dot/"
dotDir="$dbroot/$dbname/dot"
if [[ $start -le $step3 ]]; then
    for i in $(seq 0 $((numOfThreads-1))); do
        args="-p $pesudoDir -dot $dotDir -n $numOfThreads -i $i"
        echo "getDotFiles $args $((i+1)) &"
        getDotFiles "$args" "$((i+1))" &
    done

    # wait
    waitFlagFiles "$numOfThreads" "getDotFiles"

    # find all finished files and delete them; '\;' indicates the command ends.
    find /tmp -maxdepth 1 -type f -iname 'getDotFiles_finished?' -exec rm {} \;
    echo "getDotFiles done"
fi

# to remove error dot files
step4=$((step3 + 1))
dotDirPath=$dotDir
if [[ $start -le $step4 ]]; then
    for i in $(seq 0 $((numOfThreads-1))); do
        args="-dot $dotDirPath -p $numOfThreads -i $i"
        echo "removeErrorDots $args $((i+1)) &"
        removeErrorDots "$args" "$((i+1))" &
    done

    # wait
    waitFlagFiles "$numOfThreads" "removeErrorDots"

    # find all finished files and delete them; '\;' indicates the command ends.
    find /tmp -maxdepth 1 -type f -iname 'removeErrorDots_finished?' -exec rm {} \;
    echo "removeErrorDots done"
fi

# to get all networkx graphs for all AST & PDG
step5=$((step4 + 1))
attackedFlag=" "
if [[ $flag -ne 0 ]] && [[ ${dataset,,} == 'gcj' ]]; then
    #statements
    attackedFlag="--attacked"
fi

if [[ $start -le $step5 ]]; then
    #statements
    # on my desktop can only allow two processes at the same time
    dataPath='/media/yihua/2TB/backup/graph_models/diffpool/data'
    if [[ ${dataset,,} == 'github' ]]; then
        #statements
        arg1="-dot $dotDirPath -d $dataPath -n $dataName -bn $numOfBalance -under_sample -N 1 -i 0"
    else
        arg1="-dot $dotDirPath -d $dataPath -n $dbname -N 2 -i 0 $attackedFlag"
        arg2="-dot $dotDirPath -d $dataPath -n $dbname -N 2 -i 1 $attackedFlag"
    fi
    echo "getASTnPDG $arg1 1 &"
    echo "getASTnPDG $arg2 2 &"

    getASTnPDG "$arg1" "1" &
    getASTnPDG "$arg2" "2" &

    # wait
    while [[ ! -f /tmp/getASTnPDG_finished1 || ! -f /tmp/getASTnPDG_finished2 ]]; do
    # while [[ ! -f /tmp/getASTnPDG_finished1  ]]; do
        sleep 1
    done

    # find all finished files and delete them; '\;' indicates the command ends.
    find /tmp -maxdepth 1 -type f -iname 'getASTnPDG_finished?' -exec rm {} \;

    echo "getASTnPDG done"
fi

end_time=$(date +%s)

echo "preprocess finished in $((end_time-start_time))"