FROM python:3.8

WORKDIR mpm/

COPY MsPacman.py mpm/
COPY dependencies.txt mpm/

RUN sudo pip3 install -r mpm/dependencies.txt

CMD ["sudo", "python3", "MsPacman.py"]
