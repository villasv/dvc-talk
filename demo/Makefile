
start:
	./run 2>&1 | awk "!/+ trap/ && !/++ set/"

clean:
	rm -rf .git .dvc data code 2>/dev/null
