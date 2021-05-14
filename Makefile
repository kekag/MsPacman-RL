
PYTHON = python3

all:
	clean install gen run

install:
	sudo pip3 install -r requirements.txt

grpcinstall:
	pip3 install grpcio-tools
	pip3 install grpcio-reflection

grpcgen:
	python3 -m grpc_tools.protoc -I protos --python_out=. --grpc_python_out=. protos/mpm/*.proto

run:
	sudo ${PYTHON} MsPacman.py

clean:
	rm -rf ./mpm
	rm -rf ./google
	rm -rf ./__pycache__
