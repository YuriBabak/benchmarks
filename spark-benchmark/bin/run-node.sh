#!/usr/bin/env bash

wget http://mirror.linux-ia64.org/apache/spark/spark-2.3.1/spark-2.3.1-bin-hadoop2.7.tgz
tar -xzf spark-2.3.1-bin-hadoop2.7.tgz
./spark-2.3.1-bin-hadoop2.7/sbin/start-master.sh
