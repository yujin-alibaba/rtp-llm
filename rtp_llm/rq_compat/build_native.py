"""Build compatibility libraries in an isolated directory using current SDK/Torch.
Requires the frozen FlashInfer 0.2.6 source package, not its prebuilt old-SDK kernels.
Always replay fixed component fixtures after rebuilding (compiler flags affect arithmetic).
"""
import argparse, os, pathlib, shutil, subprocess, sys
p=argparse.ArgumentParser();p.add_argument('--flashinfer-source',required=True);p.add_argument('--output',required=True);p.add_argument('--nvcc',default='nvcc');a=p.parse_args()
out=pathlib.Path(a.output).resolve();out.mkdir(parents=True,exist_ok=False);root=pathlib.Path(__file__).parent
for name in ['legacy_norm_probe','rope-legacy']:
 subprocess.run([a.nvcc,'-O3','-shared','-Xcompiler','-fPIC',str(root/'csrc'/(name+'.cu')),'-o',str(out/(name+'.so'))],check=True)
# Modify only a private source copy; do not overwrite the installed FlashInfer.
site=out/'source';site.mkdir();shutil.copytree(pathlib.Path(a.flashinfer_source)/'flashinfer',site/'flashinfer',ignore=shutil.ignore_patterns('*.so','__pycache__','*.pyc'))
p=site/'flashinfer/jit/cpp_ext.py';s=p.read_text()
if '--static-global-template-stub=false' not in s:
 s=s.replace('cuda_cflags: List[str] = []','cuda_cflags: List[str] = ["--static-global-template-stub=false"]')
 if '--static-global-template-stub=false' not in s:raise RuntimeError('Unsupported frozen FlashInfer source layout')
 p.write_text(s)
env=os.environ.copy();env['PYTHONPATH']=str(site);env['TORCH_CUDA_ARCH_LIST']='8.0';env['MAX_JOBS']='8';env['FLASHINFER_WORKSPACE_BASE']=str(out/'cache')
code="""import pathlib,torch,shutil
from flashinfer.jit import env
from flashinfer.jit.attention.pytorch import gen_batch_decode_module
spec=gen_batch_decode_module(torch.float16,torch.float16,torch.float16,torch.int32,128,128,0,False,False)
spec.build_and_load()
name='batch_decode_with_kv_cache_dtype_q_f16_dtype_kv_f16_dtype_o_f16_dtype_idx_i32_head_dim_qk_128_head_dim_vo_128_posenc_0_use_swa_False_use_logits_cap_False'
shutil.copy2(env.FLASHINFER_JIT_DIR/name/(name+'.so'),pathlib.Path(OUTPUT)/(name+'.so'))
""".replace('OUTPUT',repr(str(out)))
subprocess.run([sys.executable,'-c',code],env=env,check=True)
