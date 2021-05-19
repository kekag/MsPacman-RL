PIP = pip3
PYTHON = python3

all:
	install gen run

install:
	sudo ${PIP} install -r requirements.txt

gen:
	${PYTHON} -m grpc_tools.protoc -I protos --python_out=. --grpc_python_out=. protos/*.proto

run:
	sudo ${PYTHON} MsPacman.py

server:
	sudo ${PYTHON} MsPacman.py -server

client:
	sudo ${PYTHON} MsPacman.py -client

clean:
	rm -rf ./__pycache__