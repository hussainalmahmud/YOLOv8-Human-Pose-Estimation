install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt


lint:
	pylint --disable=R,C *.py

format:
	black *.py

all: install lint test format 


