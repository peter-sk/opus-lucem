SERVER_ARGS = {
     "host": "0.0.0.0",
     "port": 8000
}

APP_ARGS = {
    "langs": "en,da,sv", #"en,da,de,es,fi,fr,no,ru,sv,zh,ar",
    "directions": "all",
    "model_templates": "Helsinki-NLP/opus-mt-tc-big-%s-%s,Helsinki-NLP/opus-mt-%s-%s",
    "num_sequences": 3,
    "device": "cpu",
    "verbosity": 4 # 0 = quiet, 1 = error, 2 = warning, 3 = info, 4 = debug
}