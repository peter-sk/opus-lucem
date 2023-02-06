import argparse
import flask
import sys
import time
import torch
import transformers


lucem = None


class Lucem:
    def __init__(self, languages, gpu=False, debug=False):
        self._gpu = gpu
        self._debug = debug
        self._MODEL_TEMPLATES = [
            "Helsinki-NLP/opus-mt-tc-big-%s-%s",
            "Helsinki-NLP/opus-mt-%s-%s"
        ]
        self._NUM_SEQUENCES = 3

        self._translators = {l1: {l2: self._translator(l1, l2) for l2 in languages if l1 != l2} for l1 in languages}
        for l1 in languages:
            for l2 in languages:
                if l1 != l2 and not self._translators[l1][l2]:
                    for l in languages:
                        if l not in (l1, l2) and len(self._translators[l1][l]) == 1 and len(self._translators[l][l2]) == 1:
                            self._translators[l1][l2] = (self._translators[l1][l][0], self._translators[l][l2][0])
                            break

        if gpu:
            print("Moving data to GPU")
            for l1, l2ts in self._translators.items():
                for l2, ts in l2ts.items():
                    l2ts[l2] = tuple((t[0],t[1].to("cuda")) for t in ts)


    def translate(self, l1, l2, l1_text):
        if self._debug:
            print("request l1 = %s, l2 = %s, l1_text = %s" % (repr(l1), repr(l2), repr(l1_text)))
        if l1 == "auto":
            l1 = langid.classify(l1_text)[0]
            if self_debug:
                print("detected l1 = %s" % repr(l1))
        if l1 == l2:
            return {
                "status" : "OK",
                "translation" : l1_text
            }
        l1_translators = self._translators.get(l1, None)
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
            device = "cuda" if self._gpu else "cpu"
            while True:
                tokenizer, model = translator[0]
                l1_tokens = tokenizer(l1_text, return_tensors='pt').to(device)
                max_length = 2*len(l1_tokens['input_ids'][0])
                l2_tokens = model.generate(**l1_tokens, max_length=max_length, num_return_sequences=self._NUM_SEQUENCES)
                l2_texts = [tokenizer.decode(l2_tokens[i], skip_special_tokens=True) for i in range(self._NUM_SEQUENCES)]
                if self._debug:
                    print("translated l1 = %s, l2 = %s, l1_text = %s, l2_texts = %s" % (repr(l1), repr(l2), repr(l1_text), repr(l2_texts)))
                translator = translator[1:]
                if not translator:
                    return {
                        "status" : "OK",
                        "translations" : l2_texts
                    }
                l1_text = l2_texts[0]
        except BaseException as e:
            return {
                "status" : "ERROR",
                "error" : "EXCEPTION",
                "message" : str(e)
            }
            if self._debug:
                raise e

    def _translator(self, l1, l2):
        print("Loading model %s -> %s" % (l1, l2))
        for model_template in self._MODEL_TEMPLATES:
            model = model_template % (l1, l2)
            try:
                tp = (transformers.AutoTokenizer.from_pretrained(model), transformers.AutoModelForSeq2SeqLM.from_pretrained(model))
                return (tp,)
            except:
                pass
        return ()


def create_app(languages=['da', 'en'], gpu=False, allow_tf32=False, allow_fp16=False, debug=False):
    global lucem

    if allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    if allow_fp16:
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True

    device_type = "cuda" if gpu else "cpu"
    if lucem is None:
        lucem = Lucem(gpu=gpu, debug=debug, languages=languages)

    app = flask.Flask(__name__)
    app.add_url_rule("/<l1>/<l2>/<l1_text>", methods=['GET'], view_func=lucem.translate)

    return app


def gunicorn_app():
    return create_app()


def main():
    parser = argparse.ArgumentParser(description='Opus Lucem server')
    parser.add_argument('--port', metavar='PORT', type=int, default=8000, help='Listen on port')
    parser.add_argument('--gpu', action='store_true', default=False, help='Use GPU')
    parser.add_argument('--allow-tf32', action='store_true', default=False, help='Allow use of TF32 (for GPU)')
    parser.add_argument('--allow-fp16', action='store_true', default=False, help='Allow use of FP16 (for GPU)')
    parser.add_argument('--debug', action='store_true', default=False, help='Enable debug messages')
    parser.add_argument('--threads', metavar='NUM', type=int, help='Override number of threads')
    parser.add_argument('languages', metavar='LANG', nargs='+', help='List of languages to load')

    args = parser.parse_args()

    thread_count = args.threads
    if thread_count is not None:
        torch.set_num_threads(thread_count)

    print("Number of threads: %d" % torch.get_num_threads())

    app = create_app(languages=args.languages, gpu=args.gpu, allow_tf32=args.allow_tf32, allow_fp16=args.allow_fp16, debug=args.debug)
    app.run(host='0.0.0.0', port=args.port)


if __name__ == '__main__':
    main()
