#!/usr/bin/env python3
import base64,io,json,time,urllib.request,concurrent.futures
from PIL import Image,ImageDraw
base='http://127.0.0.1:6022'
def chat(body):
 req=urllib.request.Request(base+'/v1/chat/completions',json.dumps(body).encode(),{'Content-Type':'application/json'})
 start=time.monotonic()
 with urllib.request.urlopen(req,timeout=300) as r:data=json.load(r)
 return {'seconds':round(time.monotonic()-start,2),'usage':data['usage'],'message':data['choices'][0]['message'],'finish':data['choices'][0]['finish_reason']}
im=Image.new('RGB',(640,400),'white');draw=ImageDraw.Draw(im);draw.rectangle((40,100,250,310),fill='red');draw.ellipse((370,100,580,310),fill='blue');b=io.BytesIO();im.save(b,format='PNG');url='data:image/png;base64,'+base64.b64encode(b.getvalue()).decode()
longtext='Read the image and find the unique archive code in this text. '+(' alpha beta gamma delta'*16250)+' The archive code is WISLA-7429-ZEFIR. '+(' alpha beta gamma delta'*16250)+' Give the code, shapes, and colors.'
r=chat({'model':'inkling','messages':[{'role':'user','content':[{'type':'text','text':longtext},{'type':'image_url','image_url':{'url':url}}]}],'max_tokens':768,'temperature':0,'reasoning_effort':'none'})
print('LONG_VISION',json.dumps(r,ensure_ascii=False),flush=True)
if r['finish']!='stop' or 'WISLA-7429-ZEFIR' not in r['message'].get('content',''): raise SystemExit('long-vision failed')
body={'model':'inkling','messages':[{'role':'user','content':'Call the function get_weather for city Warsaw.'}], 'tools':[{'type':'function','function':{'name':'get_weather','description':'Get weather for a city','parameters':{'type':'object','properties':{'city':{'type':'string'}},'required':['city']}}}], 'temperature':0,'max_tokens':512}
r2=chat(body);print('TOOL',json.dumps(r2,ensure_ascii=False),flush=True)
if not r2['message'].get('tool_calls'):raise SystemExit('tool call failed')
def long_parallel(i):
    word = [' alpha beta gamma delta', ' omega theta sigma kappa'][i]
    repeats = [15000, 10000][i]
    marker = ['SILVER-4167','COPPER-9521'][i]
    text = 'Find and repeat the unique key: ' + word * (repeats//2) + ' Unique key is '+marker+'. ' + word * (repeats-repeats//2) + ' What is the unique key?'
    res = chat({'model':'inkling','messages':[{'role':'user','content':text}],'max_tokens':768,'temperature':0,'reasoning_effort':'none'})
    res['marker']=marker
    return res
start=time.monotonic()
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
    parallel=list(pool.map(long_parallel,(0,1)))
print('PARALLEL_LONG',json.dumps({'wall_s':round(time.monotonic()-start,2),'runs':parallel},ensure_ascii=False),flush=True)
for item in parallel:
    if item['marker'] not in item['message'].get('content','') or item['finish']!='stop': raise SystemExit('parallel retrieval failed')
with open('inkling-research/basalt-long-validation.json','w') as f: json.dump({'long_vision':r,'tool':r2,'parallel_long':parallel},f,indent=2,ensure_ascii=False)
