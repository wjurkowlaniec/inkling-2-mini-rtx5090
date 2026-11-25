import base64,io,json,urllib.request
from PIL import Image,ImageDraw
im=Image.new('RGB',(640,400),'white');draw=ImageDraw.Draw(im);draw.rectangle((40,100,250,310),fill='red');draw.ellipse((370,100,580,310),fill='blue');b=io.BytesIO();im.save(b,format='PNG');url='data:image/png;base64,'+base64.b64encode(b.getvalue()).decode()
def chat(body):
 req=urllib.request.Request('http://127.0.0.1:6022/v1/chat/completions',json.dumps(body).encode(),{'Content-Type':'application/json'})
 with urllib.request.urlopen(req,timeout=300) as r:return json.load(r)
p={'model':'inkling','messages':[{'role':'user','content':[{'type':'text','text':'Describe the shapes and colors.'},{'type':'image_url','image_url':{'url':url}}]}], 'temperature':0,'max_tokens':1536,'reasoning_effort':'none'}
v=chat(p);print('VISION',json.dumps(v),flush=True)
body={'model':'inkling','messages':[{'role':'user','content':'Call the function get_weather for city Warsaw.'}], 'tools':[{'type':'function','function':{'name':'get_weather','description':'Get weather for a city','parameters':{'type':'object','properties':{'city':{'type':'string'}},'required':['city']}}}], 'temperature':0,'max_tokens':1024,'reasoning_effort':'none'}
t=chat(body);print('TOOL',json.dumps(t),flush=True)
with open('inkling-research/basalt-smoke.json','w') as f:json.dump({'vision':v,'tool':t},f,indent=2)
