#!/usr/bin/env python3
"""Small, public, task-diverse chat benchmark; no quality claims beyond explicit checks."""
import argparse, concurrent.futures, json, re, statistics, threading, time, urllib.request
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--url',required=True);p.add_argument('--output',required=True);p.add_argument('--concurrency',type=int,choices=[1,2],default=1);args=p.parse_args()
tasks=[
 ('python','Write a complete Python implementation of an in-memory TTL cache using time.monotonic, with get, set and eviction behavior. Explain complexity and include a short usage example.'),
 ('reason','A service processes 4800 jobs in 8 minutes at a constant rate. How many jobs in 15 minutes? Show the calculation and give the final integer.'),
 ('extract','From this incident note extract the exact incident ID and affected component. Note: At 10:42 UTC, incident ZX-2481 affected the payments-worker after a deployment; search-api was healthy. Return a concise answer.'),
 ('sql','Write a PostgreSQL query for daily revenue in 2025 from orders(created_at timestamptz, amount numeric, status text), counting only paid orders and ordering by day. Briefly explain the timezone choice.'),
 ('design','Design an idempotent webhook handler with retries, an outbox, error monitoring, and safeguards against duplicate messages. Give concrete implementation steps.'),
 ('json','Return only a JSON object with keys city and values. city must be Warsaw; values must contain exactly the integers 2, 3, 5 in that order.')]
def request(item,barrier):
 name,prompt=item;payload={'model':'inkling','messages':[{'role':'user','content':prompt}],'temperature':0,'max_tokens':3072,'stream':True,'stream_options':{'include_usage':True},'reasoning_effort':'none','chat_template_kwargs':{'reasoning_effort':'none'}}
 req=urllib.request.Request(args.url+'/v1/chat/completions',json.dumps(payload).encode(),{'Content-Type':'application/json'});barrier.wait();start=time.monotonic();first=None;usage={};finish=None;content=[];reason=[];done=False
 with urllib.request.urlopen(req,timeout=300) as resp:
  for raw in resp:
   line=raw.decode().strip()
   if not line.startswith('data:'):continue
   data=line[5:].strip()
   if data=='[DONE]':done=True;break
   obj=json.loads(data)
   if obj.get('usage'):usage=obj['usage']
   for c in obj.get('choices',[]):
    finish=c.get('finish_reason') or finish;d=c.get('delta') or {};a=d.get('content') or '';b=d.get('reasoning_content') or d.get('reasoning') or ''
    if (a or b) and first is None:first=time.monotonic()
    content.append(a);reason.append(b)
 end=time.monotonic();text=''.join(content);n=usage.get('completion_tokens',0);checks={
 'python':lambda:all(x in text.lower() for x in ['def get','def set','monotonic']),
 'reason':lambda:bool(re.search(r'\b9[ ,]?000\b',text)),
 'extract':lambda:'ZX-2481' in text and 'payments-worker' in text,
 'sql':lambda:all(x in text.lower() for x in ['select','orders','status','paid','group by']),
 'design':lambda:all(x in text.lower() for x in ['idempot','outbox','retr']),
 'json':lambda:json.loads(text)=={'city':'Warsaw','values':[2,3,5]}}
 try:passed=bool(checks[name]())
 except Exception:passed=False
 return {'task':name,'prompt':prompt,'start':start,'end':end,'ttft_s':round(first-start,4) if first else None,'elapsed_s':round(end-start,4),'output_tokens':n,'estimated_decode_tok_s':round(n/(end-first),2) if first and end>first else None,'output_per_total_s':round(n/(end-start),2),'finish_reason':finish,'stream_done':done,'basic_check_passed':passed,'content':text,'reasoning':''.join(reason),'usage':usage}
rows=[]
for i in range(0,len(tasks),args.concurrency):
 group=tasks[i:i+args.concurrency];barrier=threading.Barrier(len(group))
 with concurrent.futures.ThreadPoolExecutor(max_workers=len(group)) as ex:out=list(ex.map(lambda item:request(item,barrier),group))
 wall=max(r['end'] for r in out)-min(r['start'] for r in out)
 for r in out:r.pop('start');r.pop('end')
 batch={'wall_s':round(wall,4),'aggregate_tok_s':round(sum(r['output_tokens'] for r in out)/wall,2),'requests':out};rows.append(batch)
 print(json.dumps({'wall_s':batch['wall_s'],'aggregate_tok_s':batch['aggregate_tok_s'],'requests':[{k:r[k] for k in ['task','elapsed_s','output_tokens','estimated_decode_tok_s','finish_reason','basic_check_passed']} for r in out]}),flush=True)
Path(args.output).write_text(json.dumps({'url':args.url,'concurrency':args.concurrency,'method':'six public prompts, one run each, temperature 0, max 3072, EOS allowed; heuristic checks are not comprehensive quality tests. Estimated decode uses first visible chunk to end of stream; includes parser/HTTP delays.','batches':rows},indent=2)+'\n')
