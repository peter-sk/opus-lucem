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
    print("%s   " % msg,end=end)


# language and model selection
from .config import LANGS, MODEL_TEMPLATES

start("Available languages")
langs = sys.argv[1].split(",") if len(sys.argv) > 1 and "," in sys.argv[1] else LANGS
status("'%s'" % "', '".join(langs))

start("Available model templates")
model_templates = sys.argv[2].split(",") if len(sys.argv) > 2 and "," in sys.argv[1] else MODEL_TEMPLATES
status("'%s'" % "', '".join(model_templates))


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
            return tp
        except:
            pass
    status("FAILED")
    return None

translators = {l1: {l2: t for l2 in langs if l1 != l2 for t in [t for t in [translator(l1, l2)] if t]} for l1 in langs}


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
        max_length = 2*len(l1_text.split())
        l2_text = translator(l1_text, max_length=max_length)[0]['translation_text']
        return {
            "status" : "OK",
            "translation" : l2_text
        }
    except BaseException as e:
        return {
            "status" : "ERROR",
            "error" : "EXCEPTION",
            "debug" : """<table><tr><th>L1</th><td>%s</td></tr>
<tr><th>L2</th><td>%s</td></tr>
<tr><th>L1_text</th><td>%s</td></tr>
</table>
""" % (l1, l2, l1_text),
            "message" : str(e)
        }
