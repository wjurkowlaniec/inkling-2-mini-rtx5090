#!/usr/bin/env python3
"""Streaming TTFT and decode benchmark, adapted from local-models/benchmark_generation.py.

Uses SGLang /tokenize, avoiding a local tokenizer dependency. Fresh random prompt
prefixes prevent prefix-cache reuse across trials. Times include network and queueing.
"""
import argparse
import json
import secrets
import statistics
import time
import urllib.request
from pathlib import Path


def post_json(url, body, timeout=900):
    req = urllib.request.Request(url, json.dumps(body).encode(), {'Content-Type':'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def prompt_for_size(base, target):
    prefix = 'Benchmark run ' + secrets.token_hex(16) + ' '
    filler = ' alpha beta gamma delta'
    # Four tokens per repetition for this tokenizer; record exact count below.
    prompt = prefix + filler * max(1, (target - 12) // 4)
    count = post_json(base + '/tokenize', {'model':'inkling','prompt':prompt})['count']
    return prompt, count


def trial(base, target, output_tokens):
    prompt, content_count = prompt_for_size(base, target)
    payload = {'model':'inkling','messages':[{'role':'user','content':prompt}],
               'temperature':0, 'max_tokens':output_tokens, 'min_tokens':output_tokens,
               'ignore_eos':True, 'stream':True, 'stream_options':{'include_usage':True},
               'chat_template_kwargs':{'reasoning_effort':'none'}}
    req = urllib.request.Request(base+'/v1/chat/completions', json.dumps(payload).encode(),
                                 {'Content-Type':'application/json'})
    start = time.monotonic(); first = last = None; usage = {}; finish = None; done = False
    with urllib.request.urlopen(req, timeout=900) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith('data:'): continue
            data = line[5:].strip()
            if data == '[DONE]': done = True; break
            obj = json.loads(data)
            if obj.get('usage'): usage = obj['usage']
            for choice in obj.get('choices', []):
                finish = choice.get('finish_reason') or finish
                delta = choice.get('delta') or {}
                if any(delta.get(key) for key in ('content','reasoning_content','reasoning')):
                    now = time.monotonic()
                    if first is None: first = now
                    last = now
    elapsed = time.monotonic()-start
    if first is None or not usage or not done:
        raise RuntimeError('Stream incomplete or missing timing/usage')
    output = usage.get('completion_tokens', 0)
    visible_decode = (last-first) if last is not None else 0
    # Count through end of stream: the reasoning parser can buffer output.
    decode = elapsed - (first-start)
    return {'target_content_tokens':target,'actual_content_tokens':content_count,
            'usage':usage,'ttft_s':round(first-start,4),'total_s':round(elapsed,4),
            'visible_decode_s':round(visible_decode,4), 'first_to_end_s':round(decode,4),
            'estimated_output_tokens_per_s':round(output/decode,2) if decode > 0 else None,
            'finish_reason':finish,'stream_done':done}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--url',default='http://127.0.0.1:6020')
    p.add_argument('--sizes',type=int,nargs='+',default=[1000,16000,30000,60000])
    p.add_argument('--iterations',type=int,default=3)
    p.add_argument('--output-tokens',type=int,default=512)
    p.add_argument('--results',default='benchmark_generation.local.json')
    args=p.parse_args(); base=args.url.rstrip('/')
    print('Warmup...',flush=True); trial(base,1000,64)
    runs=[]
    for target in args.sizes:
        for i in range(args.iterations):
            row=trial(base,target,args.output_tokens); row['iteration']=i+1; runs.append(row)
            print(json.dumps(row),flush=True)
    summaries=[]
    for size in args.sizes:
        a=[r for r in runs if r['target_content_tokens']==size]
        summaries.append({'target_content_tokens':size,
           'actual_prompt_tokens_median':statistics.median(r['usage']['prompt_tokens'] for r in a),
           'ttft_s_median':round(statistics.median(r['ttft_s'] for r in a),3),
           'estimated_decode_tok_s_median':round(statistics.median(r['estimated_output_tokens_per_s'] for r in a if r['estimated_output_tokens_per_s'] is not None),2),
           'completion_tokens':[r['usage']['completion_tokens'] for r in a]})
    result={'method':'single-request streaming, uncached randomized user prefix, temperature 0; TTFT includes queue/prefill; decode includes stream-end overhead and is an estimate',
            'url':base,'iterations':args.iterations,'output_tokens_requested':args.output_tokens,
            'summaries':summaries,'runs':runs}
    Path(args.results).write_text(json.dumps(result,indent=2)+'\n')
    print('SUMMARY',json.dumps(summaries),flush=True)

if __name__=='__main__': main()
