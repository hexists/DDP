#!/bin/bash


pid=`ps -ef | grep "port 7788" | grep -v "grep" | cut -d " " -f3`
if [ $pid == ""]; then
    pid=`ps -ef | grep "port 7788" | grep -v "grep" | cut -d " " -f2`
fi

kill $pid

graphid=`ls -alrt runs/ | grep -v -E "\.|합계" | tail -1 | awk -F' ' '{ print $NF }'`

~/.venv/bin/tensorboard --logdir ./runs/$graphid/summaries --port 7788 &

echo $graphid
