
start:
	./run 2>&1 | awk "!/+ trap/ && !/++ set/"

debug:
	./run 2>&1

clean:
	rm -rf .git .dvc data code Dvcfile 2>/dev/null
