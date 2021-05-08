FROM python:3.8

WORKDIR mpm/

COPY MsPacman.py mpm/
COPY dependencies.txt mpm/

RUN pip3 install -r mpm/dependencies.txt

CMD ["python3", "MsPacman.py"]
