from sys import argv

from . import Lucem
from .config import APP_ARGS, SERVER_ARGS

if __name__ == "__main__":
     app_args = APP_ARGS
     app_args.update({x:y for x,y in (arg.split("=") for arg in argv[1:]) if x in APP_ARGS.keys()})
     server_args = SERVER_ARGS
     server_args.update({x:y for x,y in (arg.split("=") for arg in argv[1:]) if x in SERVER_ARGS.keys()})
     Lucem(**app_args).app.run(**server_args)
