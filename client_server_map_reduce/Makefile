clean:
	rm client/client
	rm -rf client/log
	rm -rf client/MapperInput
	rm server/server

update:
	gcc -std=c99 -Wall -pthread -o server/server server/src/server.c
	gcc -Wall -pthread -g -o client/client client/src/client.c client/src/phase1.c

client:
	gcc -Wall -pthread -g -o client/client client/src/client.c client/src/phase1.c

server:
	gcc -std=c99 -Wall -pthread -o server/server server/src/server.c
