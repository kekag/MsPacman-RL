FROM python:3.8

RUN sudo pip3 install -r dependencies

CMD ["sudo", "python3", "MsPacman.py"]