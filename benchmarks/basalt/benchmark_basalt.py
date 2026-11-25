#!/usr/bin/env python3
"""Streaming TTFT and decode benchmark, adapted from local-models/benchmark_generation.py.

Uses Basalt /tokenize, avoiding a local tokenizer dependency. Fresh random prompt
prefixes prevent prefix-cache reuse across trials. Times include network and queueing.
"""
import argparse
import concurrent.futures
import json
import secrets
import statistics
import threading
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
    count = len(post_json(base + '/tokenize', {'content':prompt, 'parse_special':False})['tokens'])
    return prompt, count


def trial(base, target, output_tokens, prepared=None, barrier=None):
    prompt, content_count = prepared if prepared is not None else prompt_for_size(base, target)
    payload = {'model':'inkling','messages':[{'role':'user','content':prompt}],
               'temperature':0, 'max_tokens':output_tokens, 'stream':True, 'stream_options':{'include_usage':True},
               'reasoning_effort':'none'}
    req = urllib.request.Request(base+'/v1/chat/completions', json.dumps(payload).encode(),
                                 {'Content-Type':'application/json'})
    if barrier is not None:
        barrier.wait(timeout=60)
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
            'finish_reason':finish,'stream_done':done, '_start':start, '_end':start+elapsed}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--url',default='http://127.0.0.1:6020')
    p.add_argument('--sizes',type=int,nargs='+',default=[1000,16000,30000,60000])
    p.add_argument('--iterations',type=int,default=3)
    p.add_argument('--concurrency',type=int,default=1,help='Number of simultaneous requests; server slot limit still applies')
    p.add_argument('--output-tokens',type=int,default=512)
    p.add_argument('--results',default='benchmark_generation.local.json')
    args=p.parse_args(); base=args.url.rstrip('/')
    if args.concurrency < 1 or args.iterations < 1:
        p.error('--concurrency and --iterations must be positive')
    print('Warmup...',flush=True); trial(base,1000,64)
    runs=[]; batches=[]
    for target in args.sizes:
        for i in range(args.iterations):
            prepared=[prompt_for_size(base,target) for _ in range(args.concurrency)]
            barrier=threading.Barrier(args.concurrency)
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                futures=[pool.submit(trial,base,target,args.output_tokens,prompt,barrier) for prompt in prepared]
                rows=[future.result() for future in futures]
            wall=max(r['_end'] for r in rows)-min(r['_start'] for r in rows)
            batch={'target_content_tokens':target,'iteration':i+1,'concurrency':args.concurrency,
                   'wall_s':round(wall,4),
                   'aggregate_completion_tok_s':round(sum(r['usage']['completion_tokens'] for r in rows)/wall,2)}
            batches.append(batch)
            for j,row in enumerate(rows):
                row.pop('_start'); row.pop('_end'); row['iteration']=i+1; row['request_index']=j+1; runs.append(row)
                print(json.dumps(row),flush=True)
            print('BATCH',json.dumps(batch),flush=True)
    summaries=[]
    for size in args.sizes:
        a=[r for r in runs if r['target_content_tokens']==size]
        summaries.append({'target_content_tokens':size,
           'actual_prompt_tokens_median':statistics.median(r['usage']['prompt_tokens'] for r in a),
           'ttft_s_median':round(statistics.median(r['ttft_s'] for r in a),3),
           'estimated_decode_tok_s_median':round(statistics.median(r['estimated_output_tokens_per_s'] for r in a if r['estimated_output_tokens_per_s'] is not None),2),
           'completion_tokens':[r['usage']['completion_tokens'] for r in a],
           'aggregate_completion_tok_s_median':round(statistics.median(b['aggregate_completion_tok_s'] for b in batches if b['target_content_tokens']==size),2),
           'batch_wall_s_median':round(statistics.median(b['wall_s'] for b in batches if b['target_content_tokens']==size),3)})
    result={'method':'OpenAI chat streaming, uncached randomized user prefix, temperature 0; EOS allowed, output lengths vary; TTFT includes queue/prefill; decode includes stream-end overhead and is an estimate',
            'url':base,'iterations':args.iterations,'concurrency':args.concurrency,'output_tokens_requested':args.output_tokens,
            'summaries':summaries,'batches':batches,'runs':runs}
    Path(args.results).write_text(json.dumps(result,indent=2)+'\n')
    print('SUMMARY',json.dumps(summaries),flush=True)

if __name__=='__main__': main()
