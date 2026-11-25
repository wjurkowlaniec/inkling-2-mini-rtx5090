#!/usr/bin/env python3
"""Measure single-request generation at a fixed output budget."""
import json, sys, time, urllib.request
mode=sys.argv[1] if len(sys.argv)>1 else 'dspark'
prompt='Write a detailed technical explanation of reliable task queues, covering retries, idempotency, monitoring, and failures. Include examples.'
body={'text':prompt,'sampling_params':{'temperature':0,'max_new_tokens':1024,'ignore_eos':True}}
for i in range(3):
    req=urllib.request.Request('http://127.0.0.1:6020/generate',json.dumps(body).encode(),{'Content-Type':'application/json'})
    start=time.monotonic()
    with urllib.request.urlopen(req,timeout=300) as response: result=json.load(response)
    elapsed=time.monotonic()-start
    record={'mode':mode,'trial':i,'elapsed_s':round(elapsed,3),'meta':result.get('meta_info',{})}
    with open('speed_results.local.jsonl','a') as f: f.write(json.dumps(record)+'\n')
    print(json.dumps(record),flush=True)
