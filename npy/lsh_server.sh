#!/bin/bash
procname='lsh_server.py'
do_stop(){

    PROCESS=`ps -ef|grep ${procname}|grep -v grep|grep -v PPID|awk '{ print $2}'`
    for i in $PROCESS
    do
      echo "Kill the ${procname} process [ $i ]"
      kill -9 $i
    done

}
start(){
        python ${procname} -f features.npy -l labels.npy -k 5
        PROCESS=$(ps aux | grep ${procname} | grep -v grep )
        for i in $PROCESS
        do
          echo "start the ${procname} process [ $i ]"
        done
}
case $1 in
    start|s)
        python ${procname} -f features.npy -l labels.npy -k 5
        PROCESS=$(ps aux | grep ${procname} | grep -v grep )
        for i in $PROCESS
        do
          echo "start the ${procname} process [ $i ]"
        done
        ;;
    restart)
        do_stop
        start
        ;;
    stop)
        do_stop
        ;;
    clear)
        find . -name "*.pyc" -type f -print -exec rm -rf {} \;
        ;;
    status)
        ps -aux | grep ${procname}
        ;;
    *)

		echo "################################Usage############################ "
    echo "Usage: $0 start|stop|clear|status" >&2
        exit 3
    ;;
esac
exit 0