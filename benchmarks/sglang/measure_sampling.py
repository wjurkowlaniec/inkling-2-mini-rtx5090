#!/usr/bin/env python3
"""Compare decoding for ordinary short tasks at two sampling temperatures.

Uses /generate (raw completion), includes prefill and request overhead, and caps
output instead of forcing it; no claims about answer quality or chat behavior.
"""
import json, statistics, time, urllib.request
from pathlib import Path
url='http://127.0.0.1:6020/generate'
prompts={
 'prose':'Explain the tradeoffs between message queues and direct HTTP calls for a small engineering team. Include practical examples.\n\n',
 'code':'Write a Python function that implements a thread-safe LRU cache with tests and explain edge cases.\n\n',
 'analysis':'Compare the costs and benefits of PostgreSQL indexes for a table with a million records and frequent writes. Give a specific example.\n\n',
}
rows=[]
for temp in (0, 0.7):
 for task,prompt in prompts.items():
  for i in range(3):
   params={'temperature':temp,'max_new_tokens':512,'ignore_eos':False,'sampling_seed':1200+i}
   body={'text':prompt,'sampling_params':params}
   req=urllib.request.Request(url,json.dumps(body).encode(),{'Content-Type':'application/json'})
   t=time.monotonic()
   with urllib.request.urlopen(req,timeout=180) as resp: data=json.load(resp)
   elapsed=time.monotonic()-t; meta=data['meta_info']; n=meta['completion_tokens']
   row={'temperature':temp,'task':task,'sampling_seed':1200+i,'elapsed_s':round(elapsed,4),'prompt_tokens':meta['prompt_tokens'],'output_tokens':n,'output_per_total_s':round(n/elapsed,2),'accept_length':meta.get('spec_accept_length'),'finish_reason':meta.get('finish_reason')}
   rows.append(row);print(json.dumps(row),flush=True)
summary={f'{task}-temp-{temp}':round(statistics.median(x['output_per_total_s'] for x in rows if x['task']==task and x['temperature']==temp),2) for temp in (0,0.7) for task in prompts}
Path('sampling-benchmark-results.json').write_text(json.dumps({'method':'Raw /generate; output tokens / total request time, including prefill; three trials; EOS allowed; no forced minimum. Independent repeated requests may hit prefix cache.','prompts':prompts,'results':rows,'summary':summary},indent=2)+'\n')
print('SUMMARY',json.dumps(summary),flush=True)
