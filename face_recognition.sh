#!/bin/bash
help_info ()
{
	echo "****************************************************************************"
	echo "*"
	echo "* MODULE:              python Script - face_recognition"
	echo "*"
	echo "* COMPONENT:          This script used to run face_recognition_main python process"
	echo "*"
	echo "* REVISION:           $Revision: 1.0 $"
	echo "*"
	echo "* DATED:              $Date: 2018-07-26 15:16:28 +0000 () $"
	echo "*"
	echo "* AUTHOR:             yanhong.jia"
	echo "*"
	echo "***************************************************************************"
	echo ""
	echo "* Copyright yanhong.jia@kuang-chi.com. 2020. All rights reserved"
	echo "*"
	echo "***************************************************************************"
}

procname='face_recognition_main.py'
do_stop(){
   
    PROCESS=`ps -ef|grep ${procname}|grep -v grep|grep -v PPID|awk '{ print $2}'`
    for i in $PROCESS
    do
      echo "Kill the ${procname} process [ $i ]"
      kill -9 $i
    done

}

case $1 in
    start|s)
        python ${procname}
        PROCESS=$(ps aux | grep ${procname} | grep -v grep )
        for i in $PROCESS
        do
          echo "start the ${procname} process [ $i ]"
        done
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
		help_info
		echo "################################Usage############################ "
    echo "Usage: $0 start|stop|clear|status" >&2
        exit 3
    ;;
esac
exit 0