#!/usr/bin/env python3
"""Compare the same JSON task with and without response_format=json_object.

Uses Basalt's non-streaming timings for engine decode and client wall time.
Output validity is checked independently; JSON mode uses server grammar.
"""
import argparse
import concurrent.futures
import json
import statistics
import time
import urllib.request
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--url', default='http://127.0.0.1:6020')
p.add_argument('--iterations', type=int, default=3)
p.add_argument('--concurrency', type=int, choices=(1, 2), default=1)
p.add_argument('--output', default='structured-output.local.json')
p.add_argument('--input-comparison', action='store_true', help='Compare the same inventory as text vs JSON input, always with json_object output')
p.add_argument('--schema-comparison', action='store_true', help='Compare json_object and strict json_schema on the same tasks')
args = p.parse_args()
if args.iterations < 1:
    p.error('--iterations must be positive')
if args.input_comparison and args.schema_comparison:
    p.error('choose only one comparison')

records = [{'id': f'P{i:03d}', 'quantity': (i * 7) % 31, 'category': ('books', 'tools', 'food')[i % 3]} for i in range(1, 25)]
notes = '\n'.join(f"{r['id']}: {r['quantity']} units; category {r['category']}" for r in records)
tasks = {
    'extract': ('Convert the inventory note below to exactly one JSON object with a single key "items", '
                'whose value is an array of objects with keys "id", "quantity", and "category". '
                'Include every line in order, preserving the exact values. Output JSON only.\n\n' + notes),
    'generate': ('Generate exactly 24 fictional catalog records as one JSON object with key "items". '
                 'Each item must contain "id" (P001 through P024 in order), "quantity" (integer 0 to 30), '
                 'and "category" (books, tools, or food). Output JSON only.'),
}

if args.input_comparison:
    instruction = ('Convert the inventory below to exactly one JSON object with a single key "items", '
                   'whose value is an array of objects with keys "id", "quantity", and "category". '
                   'Include every record in order, preserving the exact values. Output JSON only.\n\n')
    tasks = {'text_input': instruction + notes,
             'json_input': instruction + json.dumps(records, separators=(',', ':'))}

def validate(text, task):
    try:
        obj = json.loads(text)
        items = obj['items']
        if set(obj) != {'items'} or not isinstance(items, list) or len(items) != 24:
            return False
        if any(set(v) != {'id','quantity','category'} or v['id'] != f'P{i:03d}' or type(v['quantity']) is not int or not 0 <= v['quantity'] <= 30 or v['category'] not in ('books','tools','food') for i,v in enumerate(items, 1)):
            return False
        return task == 'generate' or items == records
    except (ValueError, KeyError, TypeError):
        return False

def call(task, mode, temperature, seed):
    payload = {'model':'inkling', 'messages':[{'role':'user','content':tasks[task]}], 'temperature':temperature,
               'seed':seed, 'max_tokens':3072, 'reasoning_effort':'none', 'stream':False}
    if mode == 'json_object':
        payload['response_format'] = {'type':'json_object'}
    elif mode == 'json_schema':
        item = {'type':'object', 'properties':{'id':{'type':'string'}, 'quantity':{'type':'integer'}, 'category':{'type':'string','enum':['books','tools','food']}}, 'required':['id','quantity','category'], 'additionalProperties':False}
        schema = {'type':'object', 'properties':{'items':{'type':'array','items':item}}, 'required':['items'], 'additionalProperties':False}
        payload['response_format'] = {'type':'json_schema','json_schema':{'name':'inventory','strict':True,'schema':schema}}
    req = urllib.request.Request(args.url.rstrip('/')+'/v1/chat/completions', json.dumps(payload).encode(), {'Content-Type':'application/json'})
    start = time.monotonic()
    with urllib.request.urlopen(req, timeout=300) as response:
        obj = json.load(response)
    end = time.monotonic()
    choice = obj['choices'][0]
    text = choice['message'].get('content') or ''
    timings = obj.get('timings') or {}
    row = {'task':task, 'mode':mode, 'temperature':temperature, 'seed':seed,
           'elapsed_s':round(end-start,4), 'output_tokens':obj['usage']['completion_tokens'],
           'prompt_tokens':obj['usage']['prompt_tokens'], 'finish_reason':choice.get('finish_reason'),
           'valid_expected_json':validate(text, task), 'engine_timings':timings, 'content':text,
           'reasoning_chars':len(choice['message'].get('reasoning_content') or ''), '_start':start, '_end':end}
    return row

runs=[]
# Both modes use exactly the same prompt; alternate order to reduce drift.
for temp in (0, 0.7):
    for task in tasks:
        for iteration in range(args.iterations):
            modes = ('json_object',) if args.input_comparison else ((('json_object','json_schema') if iteration % 2 == 0 else ('json_schema','json_object')) if args.schema_comparison else (('plain','json_object') if iteration % 2 == 0 else ('json_object','plain')))
            for mode in modes:
                with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                    futures=[pool.submit(call,task,mode,temp,4200+iteration*10+i) for i in range(args.concurrency)]
                    batch=[f.result() for f in futures]
                wall=max(x['_end'] for x in batch)-min(x['_start'] for x in batch)
                for row in batch:
                    row.pop('_start');row.pop('_end');row['iteration']=iteration+1
                    row['batch_wall_s']=round(wall,4)
                    row['batch_output_tok_s']=round(sum(r['output_tokens'] for r in batch)/wall,2)
                    runs.append(row)
                print(json.dumps({'task':task,'mode':mode,'temperature':temp,'iteration':iteration+1,'wall_s':round(wall,3),'out':[x['output_tokens'] for x in batch],'decode':[x['engine_timings'].get('predicted_per_second') for x in batch],'valid':[x['valid_expected_json'] for x in batch],'finish':[x['finish_reason'] for x in batch]}),flush=True)
summary=[]
for temp in (0, 0.7):
 for task in tasks:
  for mode in (('json_object',) if args.input_comparison else (('json_object','json_schema') if args.schema_comparison else ('plain','json_object'))):
   rows=[r for r in runs if r['temperature']==temp and r['task']==task and r['mode']==mode]
   rates=[r['engine_timings']['predicted_per_second'] for r in rows if r['engine_timings'].get('predicted_per_second') is not None]
   summary.append({'task':task,'mode':mode,'temperature':temp,'requests':len(rows),'valid':sum(r['valid_expected_json'] for r in rows), 'finished':sum(r['finish_reason']=='stop' for r in rows), 'median_decode_tok_s':round(statistics.median(rates),2) if rates else None,'median_output_tokens':statistics.median(r['output_tokens'] for r in rows),'median_request_wall_s':round(statistics.median(r['elapsed_s'] for r in rows),3),'median_batch_output_tok_s':round(statistics.median(r['batch_output_tok_s'] for r in rows),2)})
result={'method':('Same inventory as text or JSON input; response_format=json_object.' if args.input_comparison else ('Same prompts: json_object vs strict json_schema.' if args.schema_comparison else 'Same prompts: prompt-only JSON vs response_format=json_object.')) + ' Seeded temperature 0 and 0.7; natural EOS; 3072 max. Basalt predicted_per_second is request-level decode; batch includes prefill. Validation requires exact inventory contents except generated data. Reasoning is included in output tokens.','concurrency':args.concurrency,'iterations':args.iterations,'tasks':tasks,'runs':runs,'summary':summary}
Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
print('SUMMARY',json.dumps(summary),flush=True)
