import sys
import time

# progress
started = 0
def start(*msg):
    global started
    if msg:
        print(" ".join(map(str,msg)).ljust(60),"... ",end='',flush=True)
    started = time.time()
def end(end='\n'):
    global started
    print(" %.3f seconds   " % (time.time()-started),end=end,flush=True)
    started = time.time()
def status(msg,end='\n'):
    print("%s   " % msg,end=end,flush=True)


# language and model selection
from .config import LANGS, MODEL_TEMPLATES

start("Available languages")
langs = sys.argv[1].split(",") if len(sys.argv) > 1 and "," in sys.argv[1] else LANGS
status(" ".join(langs))

start("Available model templates")
model_templates = sys.argv[2].split(",") if len(sys.argv) > 2 and "," in sys.argv[1] else MODEL_TEMPLATES
status(" ".join(model_templates))


# load libraries
start("Loading tensorflow library")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
end()
start("Loading transformers library")
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
from transformers import pipeline, logging
end()
start("Loading flask library")
from flask import Flask
end()


# model building
def translator(l1, l2):
    start("Loading model for %s --> %s" % (l1, l2))
    for model_template in model_templates:
        model = model_template % (l1, l2)
        status(model, end='')
        try:
            tp = pipeline("translation", model=model)
            end()
            return (tp,)
        except:
            pass
    status("INDIRECT")
    return ()

translators = {l1: {l2: translator(l1, l2) for l2 in langs if l1 != l2} for l1 in langs}
for l1 in langs:
    for l2 in langs:
        if l1 != l2 and not translators[l1][l2]:
            start("Indirect translation %s --> %s" % (l1, l2))
            for l in langs:
                if l not in (l1, l2) and len(translators[l1][l]) == 1 and len(translators[l][l2]) == 1:
                    translators[l1][l2] = (translators[l1][l][0], translators[l][l2][0])
                    status("%s --> %s --> %s" % (l1, l, l2))
                    break
            else:
                status("FAILED")


# web service
app = Flask(__name__)

@app.route("/<l1>/<l2>/<l1_text>", methods=['GET', 'POST'])
def hello_world(l1, l2, l1_text):
    l1_translators = translators.get(l1, None)
    if l1_translators is None:
        return {
            "status" : "ERROR",
            "error" : "L1",
            "message" : "No translators found for L1 == %s. Accessible L1 languages are '%s'." % (repr(l1), "', '".join(translators.keys()))
        }
    translator = l1_translators.get(l2, None)
    if translator is None:
        return {
            "status" : "ERROR",
            "error" : "L2",
            "message" : "No translators found for L1 == %s and L2 == %s. Accessible L2 languages for L1 == %s are '%s'." % (repr(l1), repr(l2), repr(l1), "', '".join(l1_translators.keys()))
        }
    try:
        while True:
            max_length = 2*len(l1_text.split())
            l2_text = translator[0](l1_text, max_length=max_length)[0]['translation_text']
            translator = translator[1:]
            if not translator:
                return {
                    "status" : "OK",
                    "translation" : l2_text
                }
            l1_text = l2_text
    except BaseException as e:
        return {
            "status" : "ERROR",
            "error" : "EXCEPTION",
            "message" : str(e)
        }
