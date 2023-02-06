from time import time

# progress
started = 0
def start(*msg):
    if _verbosity >= 3:
        global started
        if msg:
            print(" ".join(map(str,msg)).ljust(60),"... ",end='',flush=True)
        started = time()
def end(end='\n'):
    if _verbosity >= 3:
        global started
        print(" %.3f seconds   " % (time()-started),end=end,flush=True)
        started = time()
def status(msg,end='\n'):
    if _verbosity >= 3:
        print("%s   " % msg,end=end,flush=True)


# load libraries
def init_libraries(langs):
    global is_available, AutoTokenizer, AutoModelForSeq2SeqLM, Flask, classify
    start("Loading tensorflow library")
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import tensorflow
    end()
    start("Loading torch library")
    from torch.cuda import is_available
    end()
    start("Loading transformers library")
    os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    end()
    start("Loading flask library")
    from flask import Flask
    end()
    start("Loading langid library")
    from langid import set_languages, classify
    set_languages(langs)
    end()
    return 

# model building
def init_translator(l1, l2, model_templates):
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

def init_translators(model_templates, directions):
    translators = {l1: {l2: init_translator(l1, l2, model_templates) for l2 in ds} for l1, ds in directions.items()}
    for l1, ds in directions.items():
        for l2 in ds:
            if not translators[l1][l2]:
                start("Indirect translation %s --> %s" % (l1, l2))
                for l in directions[l1]:
                    if l2 in directions[l] and len(translators[l1][l]) == 1 and len(translators[l][l2]) == 1:
                        translators[l1][l2] = (translators[l1][l][0], translators[l][l2][0])
                        status("%s --> %s --> %s" % (l1, l, l2))
                        break
                else:
                    status("FAILED")
    return translators


# configuration
def init_languages(langs):
    start("Available languages")
    langs = langs.split(",")
    status(",".join(langs))
    return langs

def init_model_templates(model_templates):
    start("Available model templates")
    model_templates = model_templates.split(",")
    status(",".join(model_templates))
    return model_templates

def init_num_sequences(num_sequences):
    start("Number of translations to return")
    num_sequences = int(num_sequences)
    status(num_sequences)
    return num_sequences

def init_directions(directions, langs):
    start("Directions of translations")
    directions = {l1: [l2 for l2 in langs if l1 != l2] for l1 in langs} if directions == "all" else {ds[0]: [l2 for l2 in ds[1:] if ds[0] != l2] for l1l2s in directions.split(":") for ds in [l1l2s.split(",")]}
    status(":".join([",".join([l1]+ds) for l1, ds in directions.items()]))
    return directions

def init_verbosity(verbosity):
    global _verbosity
    _verbosity = int(verbosity)
    start("Verbosity level")
    status(_verbosity)

def init_device(device, translators):
    start("Determining device to run inference on")
    if not is_available():
        device = "cpu"
    status(device)
    if device != "cpu":
        start("Moving models to %s" % device)
        for l1, l2ts in translators.items():
            for l2, ts in l2ts.items():
                l2ts[l2] = tuple((t[0],t[1].to(device)) for t in ts)
        end()
    return device, translators


# web service
def ret(result):
    if _verbosity >= 4:
        print("result = %s", repr(result))
    return result

def init_app(device, langs, model_templates, directions, num_sequences, verbosity):
    init_verbosity(verbosity)
    langs = init_languages(langs)
    directions = init_directions(directions, langs)
    num_sequences = init_num_sequences(num_sequences)
    model_templates = init_model_templates(model_templates)
    init_libraries(langs)
    translators = init_translators(model_templates, directions)
    device, translators = init_device(device, translators)

    app = Flask(__name__)
    @app.route("/<l1>/<l2>/<l1_text>", methods=['GET', 'POST'])
    def translate(l1, l2, l1_text):
        if _verbosity >= 4:
            print("request l1 = %s, l2 = %s, l1_text = %s" % (repr(l1), repr(l2), repr(l1_text)))
        if l1 == "auto":
            l1 = langid.classify(l1_text)[0]
            if verbosity >= 4:
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
                l2_tokens = model.generate(**l1_tokens, max_length=max_length, num_return_sequences=num_sequences)
                l2_texts = [tokenizer.decode(l2_ts, skip_special_tokens=True) for l2_ts in l2_tokens]
                if _verbosity >= 4:
                    print("translated l1 = %s, l2 = %s, l1_text = %s, l2_texts = %s" % (repr(l1), repr(l2), repr(l1_text), repr(l2_texts)))
                translator = translator[1:]
                if not translator:
                    return ret({
                        "status" : "OK",
                        "translations" : l2_texts
                    })
                l1_text = l2_texts[0]
        except BaseException as e:
            r = ret({
                "status" : "ERROR",
                "error" : "EXCEPTION",
                "message" : str(e)
            })
            if _verbosity >= 4:
                raise e
            return r
    return app
