"""Assemble opt-in compatibility code/libraries into an already compiled RTP-LLM wheel.
No credentials, runtime installs or machine-specific paths are embedded.
"""
import argparse, base64, csv, hashlib, io, json, pathlib, shutil, zipfile
p=argparse.ArgumentParser();p.add_argument('--wheel',required=True);p.add_argument('--libraries',required=True);p.add_argument('--output',required=True);a=p.parse_args()
root=pathlib.Path(__file__).resolve().parents[2];src=pathlib.Path(a.wheel);out=pathlib.Path(a.output)/src.name
out.parent.mkdir(parents=True,exist_ok=True)
if out.exists():raise SystemExit('Refuse to overwrite wheel')
entries={}
for rel in ['rtp_llm/models_py/modules/base/cuda/norm.py','rtp_llm/models_py/modules/factory/attention/cuda_impl/base_rotary_embedding_op.py','rtp_llm/models_py/modules/factory/attention/cuda_impl/py_flashinfer_mha.py','rtp_llm/rq_compat/__init__.py','rtp_llm/rq_compat/decode.py']:
 entries[rel]=root/rel
libs=list(pathlib.Path(a.libraries).glob('*.so'))
if len(libs)!=3:raise SystemExit('Expected norm, rope and one frozen Decode library')
for f in libs:entries['rtp_llm/rq_compat/lib/'+f.name]=f
with zipfile.ZipFile(src) as z,zipfile.ZipFile(out,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as o:
 record=next(n for n in z.namelist() if n.endswith('.dist-info/RECORD'))
 rows=[row for row in csv.reader(io.StringIO(z.read(record).decode())) if row[0] not in entries and row[0]!=record]
 for info in z.infolist():
  if info.filename in entries or info.filename==record:continue
  with z.open(info) as x,o.open(info,'w',force_zip64=True) as y:shutil.copyfileobj(x,y,8*1024*1024)
 for n,f in entries.items():
  o.write(f,n);data=f.read_bytes();rows.append([n,'sha256='+base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode().rstrip('='),str(len(data))])
 rows.append([record,'','']);s=io.StringIO();csv.writer(s,lineterminator='\n').writerows(rows);o.writestr(record,s.getvalue())
h=hashlib.sha256()
with out.open('rb') as f:
 for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
manifest={'wheel':str(out),'sha256':h.hexdigest(),'size':out.stat().st_size,'compat_source_sha256':{n:hashlib.sha256(f.read_bytes()).hexdigest() for n,f in entries.items()}}
out.with_suffix('.manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest,indent=2))
