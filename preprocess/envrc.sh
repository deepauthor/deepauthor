__conda_setup="$('/home/yihua/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/yihua/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/home/yihua/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home/yihua/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
