SHELL := /bin/bash

.PHONY: clean test

venv: venv/bin/activate

venv/bin/activate: requirements.txt
	test -d venv || virtualenv venv
	. venv/bin/activate && pip install --prefer-binary -Uqr requirements.txt
	touch venv/bin/activate

test: venv
	. venv/bin/activate && pytest -s -v -x

output/manuscript.md: venv manuscript/*.md
	. venv/bin/activate && manubot process --content-directory=manuscript --output-directory=output --cache-directory=cache --skip-citations --log-level=INFO
	git remote rm rootstock

output/manuscript.html: venv output/manuscript.md
	@mkdir -p output
	. venv/bin/activate && pandoc -v \
		--defaults=./common/templates/manubot/pandoc/common.yaml \
		--defaults=./common/templates/manubot/pandoc/html.yaml

output/manuscript.docx: venv output/manuscript.md
	@mkdir -p output
	. venv/bin/activate && pandoc -v \
		--defaults=./common/templates/manubot/pandoc/common.yaml \
		--defaults=./common/templates/manubot/pandoc/docx.yaml

clean:
	rm -rf venv output
