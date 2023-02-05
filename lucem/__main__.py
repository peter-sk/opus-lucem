from sys import argv

from . import get_app

if __name__ == "__main__":
     server_params = ["host", "port"]
     app_args = {x:y for x,y in (arg.split("=") for arg in argv[1:]) if x not in server_params}
     server_args = {"host": '0.0.0.0', "port": 8000}
     server_args.update({x:y for x,y in (arg.split("=") for arg in argv[1:]) if x in server_params})
     get_app(**app_args).run(**server_args)
