#! /bin/bash

pid=$(ps aux | grep "python test_time.py" | sed -n '1p' | awk '{print $2}')
while true; do
	m=$(sudo pmap ${pid} | tail -n 1 | awk '/[0-9]K/{print $2}')
	if [ -z "$m" ]; then
		exit 0
	fi
	echo $m >> memory.txt
	sleep 1
done
