#!/usr/bin/env python3
import concurrent.futures,json,sys,time,urllib.request,statistics,hashlib
from pathlib import Path
base=sys.argv[1] if len(sys.argv)>1 else 'http://127.0.0.1:6020'
label=sys.argv[2] if len(sys.argv)>2 else 'baseline'
prompts={
 'prose':'Explain how to design a reliable asynchronous job queue. Discuss retries, idempotency, dead letter queues, monitoring, and trade-offs. Write a detailed, practical tutorial.',
 'code':'Implement a production-ready Python LRU cache and a comprehensive suite of tests. Explain complexity and edge cases.',
 'long': 'Read the following technical notes and explain their main lessons. '+(' alpha beta gamma delta'*4000)+' Describe how to build reliable background task queues.'}
def call(text, temp=0, out=512):
 body={'text':text,'sampling_params':{'temperature':temp,'max_new_tokens':out,'ignore_eos':True}}
 req=urllib.request.Request(base+'/generate',json.dumps(body).encode(),{'Content-Type':'application/json'})
 t=time.monotonic()
 with urllib.request.urlopen(req,timeout=300) as r:data=json.load(r)
 elapsed=time.monotonic()-t;meta=data['meta_info'];return {'elapsed':round(elapsed,4),'prompt_tokens':meta.get('prompt_tokens'),'output_tokens':meta.get('completion_tokens'),'tok_s':round(meta.get('completion_tokens',0)/elapsed,2),'accept_length':meta.get('spec_accept_length'),'accept_rate':meta.get('spec_accept_rate'),'text_sha256':hashlib.sha256(data['text'].encode()).hexdigest(),'end':meta.get('finish_reason')}
print('warmup',flush=True);call(prompts['prose'],out=64)
runs=[]
for key in ('prose','code','long'):
 for i in range(3):
  row={'task':key,'trial':i,**call(prompts[key])};runs.append(row);print(json.dumps(row),flush=True)
t=time.monotonic()
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
 both=list(ex.map(call,(prompts['prose'],prompts['code'])))
parallel={'wall':round(time.monotonic()-t,4),'requests':both,'combined_tok_s':round(sum(x['output_tokens'] for x in both)/(time.monotonic()-t),2)}
result={'label':label,'base':base,'runs':runs,'parallel':parallel,'summary':{key:{'median_tok_s':statistics.median(r['tok_s'] for r in runs if r['task']==key),'median_accept':statistics.median(r['accept_length'] for r in runs if r['task']==key)} for key in prompts}}
Path(label+'.json').write_text(json.dumps(result,indent=2)+'\n');print('RESULT',json.dumps(result['summary']),json.dumps(parallel),flush=True)
