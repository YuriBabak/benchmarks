#!/usr/bin/env bash

set -e
source settings.sh

for SERVER in $(cat $SERVERS_FILE); do \
ssh -v ybabak@$SERVER ""
done