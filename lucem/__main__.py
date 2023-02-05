from sys import argv

from . import get_app

if __name__ == "__main__":
     get_app(**{x:y for x,y in (arg.split("=") for arg in argv[1:])}).run(host='0.0.0.0',port=8000)
