#!/usr/bin/env bash
set -e
cd ..
mvn clean compile assembly:single
cd -
bash copy.sh
bash stop-servers.sh
bash run-servers.sh
