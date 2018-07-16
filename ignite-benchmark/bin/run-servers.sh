set -e
source settings.sh

for SERVER in $(cat $SERVERS_FILE); do \
ssh aplatonov@$SERVER "bash -c '(nohup $JAVA \
-jar $WORKDIR/main.jar \
-ea \
--dataset $WORKDIR/homecredit_top10k.csv \
--cache-name HOMECREDIT \
--trainers rf -p ignite \
-m server \
--config-path $WORKDIR/server.xml &> $WORKDIR/server.log &) && pgrep java > $WORKDIR/server.pid'"; \
done
