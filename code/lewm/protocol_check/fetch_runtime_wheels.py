from pathlib import Path
import concurrent.futures,json,time,urllib.request
R=Path(__file__).parent/'wheels';R.mkdir(exist_ok=True)
specs=[('pyarrow','20.0.0','cp311-cp311-manylinux_2_28_x86_64'),('imageio-ffmpeg','0.6.0','manylinux2014_x86_64'),('wandb','0.30.0','manylinux_2_28_x86_64')]
def get(spec):
 name,version,tag=spec
 meta=json.load(urllib.request.urlopen(f'https://pypi.org/pypi/{name}/{version}/json',timeout=30))
 item=next(x for x in meta['urls'] if tag in x['filename'] and x['filename'].endswith('.whl'))
 dest=R/item['filename'];start=time.time()
 with urllib.request.urlopen(item['url'],timeout=60) as src,dest.open('wb') as out:
  while data:=src.read(1024*1024):out.write(data)
 assert dest.stat().st_size==item['size']
 return {'file':dest.name,'bytes':dest.stat().st_size,'seconds':round(time.time()-start,1)}
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
 for item in pool.map(get,specs):print(json.dumps(item),flush=True)
