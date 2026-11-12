"""Diagnostic bridge: frozen FlashInfer 0.2.6 kernel, compiled with current SDK."""
import pathlib,torch
from types import SimpleNamespace
from flashinfer.decode import BatchDecodeWithPagedKVCacheWrapper as CurrentWrapper
ROOT=pathlib.Path(__file__).parent
NAME='batch_decode_with_kv_cache_dtype_q_f16_dtype_kv_f16_dtype_o_f16_dtype_idx_i32_head_dim_qk_128_head_dim_vo_128_posenc_0_use_swa_False_use_logits_cap_False'
torch.ops.load_library(str(ROOT/'lib'/(NAME+'.so')))
MOD=getattr(torch.ops,NAME)
class LegacyWrapper(CurrentWrapper):
 def plan(self,indptr,indices,last_page_len,num_qo_heads,num_kv_heads,head_dim,page_size,**kw):
  super().plan(indptr,indices,last_page_len,num_qo_heads,num_kv_heads,head_dim,page_size,**kw)
  assert (num_qo_heads,num_kv_heads,head_dim)==(16,8,128)
  assert not self._use_tensor_cores and self._pos_encoding_mode=='NONE'
  self._plan_info=MOD.plan(self._float_workspace_buffer,self._int_workspace_buffer,self._pin_memory_int_workspace_buffer,indptr.cpu(),len(indptr)-1,num_qo_heads,num_kv_heads,page_size,self.is_cuda_graph_enabled,-1,0.,head_dim,head_dim,torch.empty(0,dtype=torch.float16),torch.empty(0,dtype=torch.float16))
  def run(*args):
   # Current API adds enable_pdl at slot13; legacy expects reciprocal RoPE arguments.
   a=list(args);a.pop(13);a[-2]=1./a[-2];a[-1]=1./a[-1];MOD.run(*a)
  self._cached_module=SimpleNamespace(run=run)
