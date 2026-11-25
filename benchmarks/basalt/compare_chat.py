#!/usr/bin/env python3
"""Same natural chat prompts and settings against OpenAI-compatible servers."""
import argparse, concurrent.futures, json, statistics, threading, time, urllib.request
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--url',required=True);p.add_argument('--output',required=True);p.add_argument('--concurrency',type=int,default=1);args=p.parse_args()
prompts=['Explain how to design a reliable asynchronous job queue. Discuss retries, idempotency, dead letter queues, monitoring, and trade-offs. Write a detailed practical tutorial.', 'Implement a production-ready Python LRU cache and a comprehensive suite of tests. Explain complexity and edge cases.']
def run(prompt,barrier):
 payload={'model':'inkling','messages':[{'role':'user','content':prompt}],'temperature':0,'max_tokens':1024,'stream':True,'stream_options':{'include_usage':True},'reasoning_effort':'none','chat_template_kwargs':{'reasoning_effort':'none'}}
 req=urllib.request.Request(args.url+'/v1/chat/completions',json.dumps(payload).encode(),{'Content-Type':'application/json'});barrier.wait();start=time.monotonic();first=None;usage={};finish=None;parts=[]
 with urllib.request.urlopen(req,timeout=300) as response:
  for raw in response:
   line=raw.decode().strip()
   if not line.startswith('data:'):continue
   data=line[5:].strip()
   if data=='[DONE]':break
   obj=json.loads(data)
   if obj.get('usage'):usage=obj['usage']
   for c in obj.get('choices',[]):
    finish=c.get('finish_reason') or finish;d=c.get('delta') or {};visible=''.join(d.get(k) or '' for k in ('content','reasoning_content','reasoning'))
    if visible:
     if first is None:first=time.monotonic()
     parts.append(visible)
 end=time.monotonic();n=usage.get('completion_tokens',0)
 return {'start':start,'end':end,'ttft_s':round(first-start,4) if first else None,'elapsed_s':round(end-start,4),'output_tokens':n,'output_per_total_s':round(n/(end-start),2),'estimated_decode_tok_s':round(n/(end-first),2) if first and end>first else None,'finish_reason':finish,'usage':usage,'text':''.join(parts)}
rows=[]
for i in range(3):
 group=[prompts[j%2] for j in range(args.concurrency)];barrier=threading.Barrier(len(group))
 with concurrent.futures.ThreadPoolExecutor(max_workers=len(group)) as ex: result=list(ex.map(lambda t:run(t,barrier),group))
 wall=max(x['end'] for x in result)-min(x['start'] for x in result)
 for j,row in enumerate(result):
  row.pop('start');row.pop('end');row['prompt']=group[j];row['iteration']=i+1
 rows.append({'requests':result,'wall_s':round(wall,4),'aggregate_tok_s':round(sum(r['output_tokens'] for r in result)/wall,2)})
 print(json.dumps({'iteration':i+1,'wall_s':round(wall,4),'aggregate_tok_s':rows[-1]['aggregate_tok_s'],'requests':[{k:v for k,v in x.items() if k not in ('text','prompt')} for x in result]}),flush=True)
Path(args.output).write_text(json.dumps({'url':args.url,'concurrency':args.concurrency,'method':'three repeats; chat completions; temperature 0; max 1024; EOS allowed; stream first visible token to stream end estimates; timings include client overhead and two-request overlap. Output may differ by engine.','batches':rows},indent=2)+'\n')
