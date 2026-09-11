"""Verify the Qwen3-8B snapshot against the saved official ModelScope manifest."""
from pathlib import Path
import argparse, json, hashlib, struct, math
from datetime import datetime, timezone
root=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--model',type=Path,default=root/'models/Qwen3-8B')
parser.add_argument('--out',type=Path,default=root/'data/interim/modelscope_qwen3_8b_verification.json')
args=parser.parse_args()
model=args.model.resolve()
remote=json.loads((root/'data/interim/modelscope_qwen3_8b_remote_manifest.json').read_text())
records=[]
for item in remote['Data']['Files']:
 if item['Type']!='blob': continue
 p=model/item['Path']
 assert p.is_file(), f'Missing {p.name}'
 assert p.stat().st_size==item['Size'], f'Size mismatch: {p.name}'
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(16*1024*1024),b''): h.update(chunk)
 assert h.hexdigest()==item['Sha256'], f'SHA256 mismatch: {p.name}'
 records.append({'file':item['Path'],'bytes':p.stat().st_size,'sha256':h.hexdigest(),'modelscope_file_revision':item['Revision']})
 print('Verified',p.name,flush=True)
config=json.loads((model/'config.json').read_text())
assert config['model_type']=='qwen3'
assert config.get('torch_dtype',config.get('dtype'))=='bfloat16'
assert not config.get('quantization_config')
index=json.loads((model/'model.safetensors.index.json').read_text())
observed={};dtypes=set();params=0;weight_bytes=0
for name in sorted(set(index['weight_map'].values())):
 p=model/name
 with p.open('rb') as f:
  header_n=struct.unpack('<Q',f.read(8))[0]
  assert header_n<20_000_000
  header=json.loads(f.read(header_n))
 end=0
 for key,v in header.items():
  if key=='__metadata__':continue
  assert key not in observed
  observed[key]=name;dtypes.add(v['dtype']);n=math.prod(v['shape']);params+=n
  assert v['dtype']=='BF16'
  lo,hi=v['data_offsets'];assert hi-lo==n*2
  weight_bytes+=hi-lo;end=max(end,hi)
 assert 8+header_n+end==p.stat().st_size
assert observed==index['weight_map']
assert weight_bytes==index['metadata']['total_size']
result={'verified_at':datetime.now(timezone.utc).isoformat(),'source':'ModelScope','model_id':'Qwen/Qwen3-8B','requested_revision':'master','local_path':str(model),'success':True,'files':records,'total_bytes':sum(r['bytes'] for r in records),'weight_shards':len(set(observed.values())),'tensor_count':len(observed),'parameter_count':params,'weight_dtypes':sorted(dtypes),'weight_payload_bytes':weight_bytes,'checks':{'all_remote_file_sizes_match':True,'all_remote_sha256_match':True,'all_indexed_tensors_present':True,'bf16_nonquantized':True,'safetensors_payload_sizes_match':True},'training_started':False}
out=args.out;out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='files'},ensure_ascii=False,indent=2))
