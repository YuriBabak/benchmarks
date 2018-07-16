set -e
source settings.sh

for SERVER in $(cat $SERVERS_FILE); do \
 scp ../target/org.apache.ignite.benchmark-1.0-SNAPSHOT-jar-with-dependencies.jar  aplatonov@$SERVER:$WORKDIR/main.jar; \
 scp ../src/main/resources/server.xml aplatonov@$SERVER:$WORKDIR/server.xml; \
 scp ../src/main/resources/client.xml aplatonov@$SERVER:$WORKDIR/client.xml; \
done
