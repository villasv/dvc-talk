
start:
	./run 2>&1 | awk "!/+ trap/ && !/++ set/"

clean:
	rm -rf .git .dvc data code Dvcfile 2>/dev/null
