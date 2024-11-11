install:
	pip install -r requirements.txt

lint:
	flake8 src/

run:
	python main.py
