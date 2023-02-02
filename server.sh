#!/bin/bash
python3 -m gunicorn -w 1 --preload lucem:app
