
venv: venv/bin/activate

venv/bin/activate: requirements.txt
	test -d venv || virtualenv venv
	. venv/bin/activate && pip install --prefer-binary -Uqr requirements.txt
	touch venv/bin/activate

test: venv
	. venv/bin/activate && pytest -s -v -x

clean:
	rm -rf venv
