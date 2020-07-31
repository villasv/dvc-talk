
start:
	./run 2>&1 | awk "!/+ trap/ && !/++ set/"

debug:
	./run 2>&1

clean:
	rm -rf .git .dvc dvc.* data code 2>/dev/null
