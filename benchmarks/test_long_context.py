#!/usr/bin/env python3
"""Exercise a long prompt and check a unique fact embedded in its middle."""
import json, time, urllib.request, sys

base='http://127.0.0.1:6020'
def post(path, payload, timeout=1800):
    req=urllib.request.Request(base+path, json.dumps(payload).encode(), {'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=timeout) as response:
        return json.load(response)

repeats=int(sys.argv[1]) if len(sys.argv)>1 else 30000
filler=' '+(' '.join(sys.argv[3].split(',')) if len(sys.argv)>3 else 'alpha beta gamma delta')
prompt=('Z materiału poniżej odczytaj kod archiwum. Odpowiedz wyłącznie kodem.\n'
        + filler*(repeats//2)
        + '\nJedyny kod archiwum to WISLA-7429-ZEFIR. Zapamiętaj go.\n'
        + filler*(repeats-repeats//2)
        + '\nJaki jest jedyny kod archiwum? Odpowiedz samym kodem.')
tokens=post('/tokenize',{'model':'inkling','prompt':prompt})['count']
print('raw_prompt_tokens',tokens,flush=True)
start=time.monotonic()
result=post('/v1/chat/completions',{'model':'inkling','messages':[{'role':'user','content':prompt}],
    'temperature':0,'max_tokens':int(sys.argv[2]) if len(sys.argv)>2 else 256,'chat_template_kwargs':{'reasoning_effort':'none'}})
elapsed=time.monotonic()-start
out={'elapsed_s':round(elapsed,2),'raw_prompt_tokens':tokens,'result':result}
with open('long_context_results.local.jsonl','a') as f: f.write(json.dumps(out,ensure_ascii=False)+'\n')
print(json.dumps(out,ensure_ascii=False),flush=True)
