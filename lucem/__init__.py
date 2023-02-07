class Lucem:

    def __init__(self, device, langs, model_templates, directions, num_sequences, verbosity):
        from time import time
        self.time = time
        self.init_verbosity(verbosity)
        self.init_languages(langs)
        self.init_directions(directions)
        self.init_num_sequences(num_sequences)
        self.init_model_templates(model_templates)
        self.init_libraries()
        self.init_translators()
        self.init_device(device)
        self.init_app()

    # progress
    def _start(self, *msg):
        if self.verbosity >= 3:
            if msg:
                print(" ".join(map(str,msg)).ljust(60),"... ",end='',flush=True)
            self.started = self.time()
    def _end(self, end='\n'):
        if self.verbosity >= 3:
            print(" %.3f seconds   " % (self.time()-self.started),end=end,flush=True)
            self.started = self.time()
    def _status(self, msg, end='\n'):
        if self.verbosity >= 3:
            print("%s   " % msg,end=end,flush=True)

    # load libraries
    def init_libraries(self):
        self._start("Loading tensorflow library")
        import os
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        import tensorflow
        self._end()
        self._start("Loading torch library")
        from torch.cuda import is_available
        self.cuda_is_available = is_available()
        self._end()
        self._start("Loading transformers library")
        os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        self.AutoTokenizer = AutoTokenizer
        self.AutoModel = AutoModelForSeq2SeqLM
        self._end()
        self._start("Loading flask library")
        from flask import Flask
        self.app = Flask(__name__)
        self._end()
        self._start("Loading langid library")
        from langid import set_languages, classify
        set_languages(self.langs)
        self.langid_classify = classify
        self._end()

    # model building
    def init_translators(self):
        def init_translator(l1, l2):
            self._start("Loading model for %s --> %s" % (l1, l2))
            for model_template in self.model_templates:
                model = model_template % (l1, l2)
                self._status(model, end='')
                try:
                    tp = (self.AutoTokenizer.from_pretrained(model), self.AutoModel.from_pretrained(model))
                    self._end()
                    return (tp,)
                except:
                    pass
            self._status("INDIRECT")
            return ()
        translators = {l1: {l2: init_translator(l1, l2) for l2 in ds} for l1, ds in self.directions.items()}
        for l1, ds in self.directions.items():
            for l2 in ds:
                if not translators[l1][l2]:
                    self._start("Indirect translation %s --> %s" % (l1, l2))
                    for l in self.directions[l1]:
                        if l2 in self.directions[l] and len(translators[l1][l]) == 1 and len(translators[l][l2]) == 1:
                            translators[l1][l2] = (translators[l1][l][0], translators[l][l2][0])
                            self._status("%s --> %s --> %s" % (l1, l, l2))
                            break
                    else:
                        self._status("FAILED")
        self.translators = translators

    # configuration
    def init_languages(self, langs):
        self._start("Available languages")
        self.langs = langs.split(",")
        self._status("langs=%s" % ",".join(self.langs))

    def init_model_templates(self, model_templates):
        self._start("Available model templates")
        self.model_templates = model_templates.split(",")
        self._status("model_templates=%s" % ",".join(self.model_templates))

    def init_num_sequences(self, num_sequences):
        self._start("Number of translations to return")
        self.num_sequences = int(num_sequences)
        self._status("num_sequences=%d" % self.num_sequences)

    def init_directions(self, directions):
        self._start("Directions of translations")
        self.directions = {l1: [l2 for l2 in self.langs if l1 != l2] for l1 in self.langs} if directions == "all" else {ds[0]: [l2 for l2 in ds[1:] if ds[0] != l2] for l1l2s in directions.split(":") for ds in [l1l2s.split(",")]}
        self._status("directions=%s" % ":".join([",".join([l1]+ds) for l1, ds in self.directions.items()]))

    def init_verbosity(self, verbosity):
        self.verbosity = int(verbosity)
        self._start("Verbosity level")
        self._status("verbosity=%d" % self.verbosity)

    def init_device(self, device):
        self._start("Determining device to run inference on")
        self.device = device if self.cuda_is_available else "cpu"
        self._status("device=%s" % self.device)
        if self.device != "cpu":
            self._start("Moving models to %s" % self.device)
            for l1, l2ts in self.translators.items():
                for l2, ts in l2ts.items():
                    l2ts[l2] = tuple((t[0],t[1].to(self.device)) for t in ts)
            self._end()

    # web service
    def init_app(self):
        @self.app.route("/<l1>/<l2>/<l1_text>", methods=['GET', 'POST'])
        def translate(l1, l2, l1_text):
            def ret(result):
                if self.verbosity >= 4:
                    print("result = %s", repr(result))
                return result
            if self.verbosity >= 4:
                print("request l1 = %s, l2 = %s, l1_text = %s" % (repr(l1), repr(l2), repr(l1_text)))
            if l1 == "auto":
                l1 = langid.classify(l1_text)[0]
                if self.verbosity >= 4:
                    print("detected l1 = %s" % repr(l1))
            if l1 == l2:
                return ret({
                    "status" : "OK",
                    "translation" : l1_text
                })
            l1_translators = self.translators.get(l1, None)
            if l1_translators is None:
                return ret({
                    "status" : "ERROR",
                    "error" : "L1",
                    "message" : "No translators found for L1 == %s. Accessible L1 languages are '%s'." % (repr(l1), "', '".join(self.translators.keys()))
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
                    l1_tokens = tokenizer(l1_text, return_tensors='pt').to(self.device)
                    max_length = 2*len(l1_tokens['input_ids'][0])
                    l2_tokens = model.generate(**l1_tokens, max_length=max_length, num_return_sequences=self.num_sequences)
                    l2_texts = [tokenizer.decode(l2_ts, skip_special_tokens=True) for l2_ts in l2_tokens]
                    if self.verbosity >= 4:
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
                if self.verbosity >= 4:
                    raise e
                return r

