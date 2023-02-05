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
from .config import LANGS, MODEL_TEMPLATES, NUM_SEQUENCES, DEBUG

start("Available languages")
#langs = sys.argv[1].split(",") if len(sys.argv) > 1 and "," in sys.argv[1] else LANGS
langs = LANGS
status(" ".join(langs))

start("Available model templates")
#model_templates = sys.argv[2].split(",") if len(sys.argv) > 2 and "," in sys.argv[1] else MODEL_TEMPLATES
model_templates = MODEL_TEMPLATES
status(" ".join(model_templates))


# load libraries
start("Loading tensorflow library")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
end()
start("Loading torch library")
import torch
end()
start("Loading transformers library")
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
end()
start("Loading flask library")
from flask import Flask
end()
start("Loading langid library")
import langid
langid.set_languages(langs)
end()

# model building
def translator(l1, l2):
    start("Loading model for %s --> %s" % (l1, l2))
    for model_template in model_templates:
        model = model_template % (l1, l2)
        status(model, end='')
        try:
            tp = (AutoTokenizer.from_pretrained(model), AutoModelForSeq2SeqLM.from_pretrained(model))
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

def get_app(cuda=None):
    global device
    start("Determining device to run inference on")
    if cuda is None:
        device = "cpu"
    else:
        if cuda == "all":
            device = "cuda"
        else:
            device = "cuda:" + cuda
    status(device)
    if cuda:
        start("Moving models to %s" % device)
        for l1, l2ts in translators.items():
            for l2, ts in l2ts.items():
                l2ts[l2] = tuple((t[0],t[1].to(device)) for t in ts)
        end()
    return app

def ret(result):
    if DEBUG:
        print("result = %s", repr(result))
    return result

@app.route("/<l1>/<l2>/<l1_text>", methods=['GET', 'POST'])
def translate(l1, l2, l1_text):
    if DEBUG:
        print("request l1 = %s, l2 = %s, l1_text = %s" % (repr(l1), repr(l2), repr(l1_text)))
    if l1 == "auto":
        l1 = langid.classify(l1_text)[0]
        if DEBUG:
            print("detected l1 = %s" % repr(l1))
    if l1 == l2:
        return ret({
            "status" : "OK",
            "translation" : l1_text
        })
    l1_translators = translators.get(l1, None)
    if l1_translators is None:
        return ret({
            "status" : "ERROR",
            "error" : "L1",
            "message" : "No translators found for L1 == %s. Accessible L1 languages are '%s'." % (repr(l1), "', '".join(translators.keys()))
        })
    translator = l1_translators.get(l2, None)
    if translator is None:
        return ret({
            "status" : "ERROR",
            "error" : "L2",
            "message" : "No translators found for L1 == %s and L2 == %s. Accessible L2 languages for L1 == %s are '%s'." % (repr(l1), repr(l2), repr(l1), "', '".join(l1_translators.keys()))
        })
    try:
        while True:
            tokenizer, model = translator[0]
            l1_tokens = tokenizer(l1_text, return_tensors='pt').to(device)
            max_length = 2*len(l1_tokens['input_ids'][0])
            l2_tokens = model.generate(**l1_tokens, max_length=max_length, num_return_sequences=NUM_SEQUENCES)
            l2_texts = [tokenizer.decode(l2_tokens[i], skip_special_tokens=True) for i in range(NUM_SEQUENCES)]
            if DEBUG:
                print("translated l1 = %s, l2 = %s, l1_text = %s, l2_texts = %s" % (repr(l1), repr(l2), repr(l1_text), repr(l2_texts)))
            translator = translator[1:]
            if not translator:
                return ret({
                    "status" : "OK",
                    "translations" : l2_texts
                })
            l1_text = l2_texts[0]
    except BaseException as e:
        return ret({
            "status" : "ERROR",
            "error" : "EXCEPTION",
            "message" : str(e)
        })
        if DEBUG:
            raise e
