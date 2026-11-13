# RQ compatibility (experimental, opt-in)

Author: 杨浩

The PG default path is unchanged. `RQ_LEGACY_NUMERICS=1` selects the verified legacy RMSNorm, QK normalization, RoPE arithmetic, no-prefix FlashAttention prefill and FlashInfer 0.2.6 Decode kernel. Current scope is checkpoint-8900: FP16, hidden 1024, 16 query heads, 8 KV heads, head dimension 128. This is not validated for other models, quantization, tensor-core Decode, prefix reuse or tensor parallelism.

Use with `RQ_LEGACY_MASKED_NORMALIZATION=1`, `RQ_LEGACY_CPU_ROPE_CACHE=1`, `RQ_LEGACY_FP16_LM_HEAD=1`. `RQ_FAST_MASK_TOPK=1` uses the TRT negative-infinity mask sentinel after CSR log-softmax; invalid outputs remain filtered. All switches default off.

The `lib/` directory is an assembly input, not committed binaries. It contains two small native kernels built from csrc and one legacy FlashInfer kernel rebuilt against the **current** SDK/Torch. Do not copy a binary built against the baseline SDK into the new runtime. The isolated wheel assembly recipe and source hashes are retained in the experiment manifest. Native libraries must be rebuilt for another target ABI/hardware.

The first portable wheel passes 3-case full ordered legal SID replay. Repeated graph-enabled concurrency 4 replay found a rare invalid SID with both finite and negative-infinity masks. **Use `ENABLE_CUDA_GRAPH=0` for this compatibility path; graph-enabled RQ compatibility is not accepted.** A 360-request graph-disabled replay preserved all ordered legal SIDs; repeated final-wheel concurrency 1/4 P99 acceptance is still pending. Do not infer production readiness from this smoke test.


`RQ_FAST_CSR_BUILD=1` enables stable radix sorting and bounded per-request vector reservation. It preserves the same CSR construction and token ordering. It is independent of the numeric compatibility switches and defaults off.

`RQ_LEGACY_BEAM_WIDTH=1` keeps the configured width for intermediate RQ CSR beam steps in non-streaming, non-PD requests with surviving candidates. It uses the finite mask sentinel when retaining slots so TopK can populate their token and parent IDs. Empty and final length-limited steps retain the usual valid-candidate filtering. The switch defaults off. Use graph-disabled replay for comparisons.

Build: first compile the native RTP-LLM wheel with the pinned internal/external source pair and PPU toolchain. Then run `python -m rtp_llm.rq_compat.build_native --flashinfer-source <frozen-0.2.6-source-root> --output <new-lib-dir>` in that toolchain. The helper is a reproducibility recipe: its outputs still require fixed-input component comparison before replacement of validated libraries. Assemble with `python -m rtp_llm.rq_compat.assemble_wheel --wheel <native-wheel> --libraries <validated-lib-dir> --output <new-wheel-dir>`. The assembler regenerates RECORD and records each injected source/library SHA256. No libraries are downloaded at inference startup.
