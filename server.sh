#!/bin/bash
python3 -m gunicorn -w 4 lucem:app
