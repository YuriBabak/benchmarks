source settings.sh

for SERVER in $(cat $SERVERS_FILE); do \
 ssh aplatonov@$SERVER "bash -c 'kill \$(cat $WORKDIR/server.pid)'"; \
done

exit 0
