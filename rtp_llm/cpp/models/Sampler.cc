#include "rtp_llm/cpp/models/Sampler.h"
#include <cstring>
#include <cstdlib>
#include "rtp_llm/cpp/utils/DebugUtils.h"
#include "rtp_llm/cpp/models/logits_processor/BaseLogitsProcessor.h"
#include "rtp_llm/cpp/models/logits_processor/LogitsProcessorStates.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#include <algorithm>
#include <exception>
#include <limits>
#include <unordered_set>
#include "rtp_llm/cpp/utils/ProfilingScope.h"

using namespace std;

namespace rtp_llm {

Sampler::Sampler(const SamplerInitParams& params):
    fixed_max_batch_size_(params.max_batch_size > 0 && params.fixed_max_batch_size),
    copy_stream_(cuda_graph::graphGetStreamFromPool(false)) {
    if (params.max_batch_size > 0) {
        allocateGreedySamplingBuffers(params.max_batch_size);
    }
}

void Sampler::allocateGreedySamplingBuffers(size_t max_batch_size) {
    waitGreedySamplingBufferEvents();
    max_batch_size_ = max_batch_size;
    auto pinned_i64 = torch::TensorOptions().dtype(torch::kInt64).pinned_memory(true);
    for (auto& slot : greedy_sampling_buffer_slots_) {
        auto& buffers                = slot.buffers;
        buffers.seed_host            = torch::empty({(int64_t)max_batch_size_}, pinned_i64);
        buffers.offset_host          = torch::empty({(int64_t)max_batch_size_}, pinned_i64);
        buffers.output_ids_ptrs_host = torch::empty({(int64_t)max_batch_size_}, pinned_i64);
        buffers.max_batch_size       = max_batch_size_;
        slot.ready_event.reset();
    }
}

void Sampler::ensureGreedySamplingBuffers(size_t batch_size) {
    if (batch_size <= max_batch_size_) {
        return;
    }
    // Fixed users fail fast on impossible batch sizes. Dynamic users wait for all
    // pending slot events before rebuilding every slot together.
    RTP_LLM_CHECK_WITH_INFO(!fixed_max_batch_size_,
                            "sampler batch size [%lu] exceeds initialized max batch size [%lu]",
                            batch_size,
                            max_batch_size_);
    RTP_LLM_LOG_INFO("grow greedy sampling buffers from batch size [%lu] to [%lu]", max_batch_size_, batch_size);
    allocateGreedySamplingBuffers(batch_size);
}

void Sampler::waitGreedySamplingBufferEvents() {
    for (auto& slot : greedy_sampling_buffer_slots_) {
        if (slot.ready_event) {
            slot.ready_event->synchronize();
            slot.ready_event.reset();
        }
    }
}

GreedySamplingBuffers& Sampler::nextGreedySamplingBuffers(size_t batch_size) {
    ensureGreedySamplingBuffers(batch_size);
    auto& slot = greedy_sampling_buffer_slots_[greedy_sampling_buffer_index_];
    if (slot.ready_event) {
        slot.ready_event->synchronize();
        slot.ready_event.reset();
    }
    current_greedy_sampling_slot_ = &slot;
    greedy_sampling_buffer_index_ = (greedy_sampling_buffer_index_ + 1) % greedy_sampling_buffer_slots_.size();
    return slot.buffers;
}

void Sampler::markGreedySamplingBufferReady() {
    if (current_greedy_sampling_slot_ != nullptr) {
        auto* slot = current_greedy_sampling_slot_;
        try {
            slot->ready_event             = runtimeCreateEvent();
            current_greedy_sampling_slot_ = nullptr;
        } catch (...) {
            current_greedy_sampling_slot_ = nullptr;
            slot->ready_event.reset();
            runtimeSyncAndCheck();
            throw;
        }
    }
}
SamplerOutput Sampler::forward(const SamplerInputs& inputs) {
    RTP_LLM_LOG_DEBUG(__PRETTY_FUNCTION__);
    RTP_LLM_PROFILE_SCOPE("sampler.forward");
    RTP_LLM_CHECK_WITH_INFO(!forward_in_progress_.exchange(true),
                            "Sampler::forward is single-threaded and must not be called concurrently or reentrantly");
    struct SamplerForwardGuard {
        std::atomic<bool>& forward_in_progress;
        ~SamplerForwardGuard() {
            forward_in_progress.store(false);
        }
    } sampler_forward_guard{forward_in_progress_};

    // Helper: narrow a tensor if defined, else return undefined tensor
    auto mayNarrow = [](const torch::Tensor& t, int64_t offset, int64_t size) -> torch::Tensor {
        return t.defined() ? t.narrow(0, offset, size) : torch::Tensor();
    };

    // Helper: convert optional tensor slice to std::optional<torch::Tensor>
    auto mayOptNarrow = [](const torch::Tensor& t, int64_t offset, int64_t size) -> std::optional<torch::Tensor> {
        return t.defined() ? std::optional<torch::Tensor>(t.narrow(0, offset, size)) : std::nullopt;
    };

    uint64_t max_seq_len   = inputs.token_ids.size(1);
    auto     num_beams_in  = inputs.num_beams_in.data_ptr<int64_t>();
    auto     num_beams_out = inputs.num_beams_out.data_ptr<int64_t>();

    bool has_num_beams = std::any_of(num_beams_in, num_beams_in + inputs.batch_size, [](auto n) { return n > 1; })
                         || std::any_of(num_beams_out, num_beams_out + inputs.batch_size, [](auto n) { return n > 1; });
    const bool requires_independent_output = !std::equal(num_beams_in, num_beams_in + inputs.batch_size, num_beams_out);

    // allocate output tensors
    // Keep success on CUDA to avoid a blocking D2H copy: the GPU sampling kernel writes success
    // directly, and callers that need CPU access should call .cpu() explicitly.
    auto all_success =
        torch::empty({(int64_t)inputs.batch_size}, torch::TensorOptions().dtype(torch::kBool).device(torch::kCUDA));
    auto all_beam_indices =
        has_num_beams ? torch::empty({(int64_t)inputs.batch_size_out}, torch::kInt32) : torch::Tensor();
#if USING_ROCM
    // ROCm: hipMemcpyAsync from pageable memory is truly async, and
    // Tensor::record_stream() rejects at::hip streams (aborts with a device
    // type check), so keep the blocking transfer here.
    auto inputs_token_ids_cuda = inputs.token_ids.to(torch::kCUDA);
#else
    torch::Tensor inputs_token_ids_cuda;
    {
        auto main_stream     = cuda_graph::graphGetCurrentStream();
        auto copy_done_event = cuda_graph::makeGraphEvent();
        {
            cuda_graph::GraphStreamGuard guard(copy_stream_);
            inputs_token_ids_cuda = inputs.token_ids.to(torch::kCUDA, /*non_blocking=*/true);
            copy_done_event.record(copy_stream_);
        }
        copy_done_event.block(main_stream);
        inputs_token_ids_cuda.record_stream(main_stream);
    }
#endif

    auto all_token_ids_out     = requires_independent_output ?
                                     torch::empty({(int64_t)inputs.batch_size_out, (int64_t)max_seq_len},
                                              torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA)) :
                                     inputs_token_ids_cuda;
    auto all_cum_log_probs_out = requires_independent_output && inputs.cum_log_probs.defined() ?
                                     torch::empty({(int64_t)inputs.batch_size_out}, torch::kFloat32) :
                                     inputs.cum_log_probs;

    struct Segment {
        size_t begin;
        size_t end;
        bool csr;
        torch::Tensor full_lse;
    };
    std::vector<Segment> segments;
    auto append_rows = [&](size_t begin, size_t count, bool csr) {
        RTP_LLM_CHECK(count > 0 && begin + count <= inputs.batch_size);
        if (!segments.empty() && segments.back().csr == csr
            && num_beams_in[segments.back().begin] == num_beams_in[begin]
            && num_beams_out[segments.back().begin] == num_beams_out[begin]) {
            segments.back().end += count;
        } else {
            segments.push_back({begin, begin + count, csr, {}});
        }
    };
    if (inputs.requests.empty()) {
        for (size_t row = 0; row < inputs.batch_size; ++row) {
            append_rows(row, 1, false);
        }
    } else {
        size_t row = 0;
        for (const auto& request : inputs.requests) {
            append_rows(row, request.count, request.rq_csr);
            row += request.count;
        }
        RTP_LLM_CHECK(row == inputs.batch_size);
    }
    // Opt-in compatibility probe for the frozen c140e65 wheel. Keep current
    // RQ full-vocabulary scoring as the default; do not silently change ranking.
    static const bool legacy_masked_normalization = [] {
        const char* value = std::getenv("RQ_LEGACY_MASKED_NORMALIZATION");
        return value != nullptr && std::strcmp(value, "1") == 0;
    }();
    for (auto& segment : segments) {
        if (segment.csr && !legacy_masked_normalization) {
            segment.full_lse = inputs.logits.narrow(0, segment.begin, segment.end - segment.begin)
                                   .to(torch::kFloat32).logsumexp(-1, true);
        }
    }
    auto processor_errors = preprocessLogits(inputs);
    std::unordered_map<size_t, int64_t> effective_widths;
    // The beam kernel requires populated token IDs for all planned slots. A finite
    // sentinel lets it fill those slots safely, then we discard them by score.
    constexpr float masked_beam_score = -1.0e20f;
    size_t from_batch_idx_out      = 0;
    auto&  greedy_sampling_buffers = nextGreedySamplingBuffers(inputs.batch_size);
    struct GreedySamplingBufferGuard {
        Sampler* sampler = nullptr;
        ~GreedySamplingBufferGuard() {
            if (sampler == nullptr) {
                return;
            }
            try {
                sampler->markGreedySamplingBufferReady();
            } catch (const std::exception& e) {
                RTP_LLM_LOG_WARNING("failed to record greedy sampling buffer event: %s", e.what());
            } catch (...) {
                RTP_LLM_LOG_WARNING("failed to record greedy sampling buffer event");
            }
        }
    } greedy_sampling_buffer_guard{this};

    for (const auto& segment : segments) {
        const auto from_batch_idx_in = segment.begin;
        const auto cur_num_beams_in = num_beams_in[from_batch_idx_in];
        const auto cur_num_beams_out = num_beams_out[from_batch_idx_in];
        const auto batch_size_in = segment.end - segment.begin;
        RTP_LLM_CHECK(cur_num_beams_in > 0 && batch_size_in % cur_num_beams_in == 0);
        const auto beam_batch_size  = batch_size_in / cur_num_beams_in;
        const auto batch_size_out   = beam_batch_size * cur_num_beams_out;
        const auto to_batch_idx_out = from_batch_idx_out + batch_size_out;

        auto success           = all_success.narrow(0, from_batch_idx_in, batch_size_in);
        auto logits            = inputs.logits.narrow(0, from_batch_idx_in, batch_size_in);
        auto token_ids_in      = inputs_token_ids_cuda.narrow(0, from_batch_idx_in, batch_size_in);
        auto token_ids_out     = all_token_ids_out.narrow(0, from_batch_idx_out, batch_size_out);
        auto input_lengths     = inputs.input_lengths.narrow(0, from_batch_idx_in, batch_size_in);
        auto sequence_lengths  = inputs.sequence_lengths.narrow(0, from_batch_idx_in, batch_size_in);
        auto cum_log_probs_in  = mayNarrow(inputs.cum_log_probs, from_batch_idx_in, batch_size_in);
        auto cum_log_probs_out = mayNarrow(all_cum_log_probs_out, from_batch_idx_out, batch_size_out);

        torch::Tensor rq_scores;
        if (segment.csr) {
            rq_scores = legacy_masked_normalization ? logits.to(torch::kFloat32).log_softmax(-1) :
                                                       logits.to(torch::kFloat32) - segment.full_lse;
            rq_scores.masked_fill_(rq_scores.isfinite().logical_not(), -std::numeric_limits<float>::infinity());
        }

        if (cur_num_beams_in == 1 && cur_num_beams_out == 1) {
            const auto decoder_batch_size = (int64_t)inputs.sequence_lengths.size(0);
            auto       sequence_lengths_in =
                (int64_t)from_batch_idx_in < decoder_batch_size ?
                          inputs.sequence_lengths.narrow(
                        0,
                        from_batch_idx_in,
                        min((int64_t)batch_size_in, decoder_batch_size - (int64_t)from_batch_idx_in)) :
                          torch::empty({0}, torch::kInt32);

            // TODO(zhangjianning.zjn): would be better to eliminate the copy
            if (cum_log_probs_out.defined() && cum_log_probs_in.defined()) {
                cum_log_probs_out.copy_(cum_log_probs_in);
            }

            auto top_k                = inputs.top_k.narrow(0, from_batch_idx_in, batch_size_in);
            auto top_p                = inputs.top_p.narrow(0, from_batch_idx_in, batch_size_in);
            auto temperature          = inputs.temperature.narrow(0, from_batch_idx_in, batch_size_in);
            auto repetition_penalty   = mayOptNarrow(inputs.repetition_penalty, from_batch_idx_in, batch_size_in);
            auto presence_penalty     = mayOptNarrow(inputs.presence_penalty, from_batch_idx_in, batch_size_in);
            auto frequency_penalty    = mayOptNarrow(inputs.frequency_penalty, from_batch_idx_in, batch_size_in);
            auto no_repeat_ngram_size = mayOptNarrow(inputs.no_repeat_ngram_size, from_batch_idx_in, batch_size_in);
            auto all_probs            = mayOptNarrow(inputs.all_probs, from_batch_idx_in, batch_size_in);
            auto do_sample            = mayOptNarrow(inputs.do_sample, from_batch_idx_in, batch_size_in);
            auto generator            = std::vector<at::Generator>{inputs.generator.begin() + from_batch_idx_in,
                                                                   inputs.generator.begin() + from_batch_idx_in + batch_size_in};

            GreedySamplingBuffers greedy_sampling_buffer_slice{
                greedy_sampling_buffers.seed_host.narrow(0, from_batch_idx_in, batch_size_in),
                greedy_sampling_buffers.offset_host.narrow(0, from_batch_idx_in, batch_size_in),
                greedy_sampling_buffers.output_ids_ptrs_host.narrow(0, from_batch_idx_in, batch_size_in),
                batch_size_in};
            GreedySamplingBuffers* greedy_sampling_buffer_ptr = &greedy_sampling_buffer_slice;

            RTP_LLM_PROFILE_SCOPE("sampler.forward.execSampleGreedy");
            auto sampling_logits = logits;
            torch::Tensor has_legal_token;
            if (segment.csr) {
                has_legal_token = rq_scores.isfinite().any(-1, true);
                sampling_logits = logits.masked_fill(has_legal_token.logical_not(), 0);
            }
            auto greedy_output = execSampleGreedy(
                {sampling_logits,
                 input_lengths,
                 sequence_lengths_in,
                 token_ids_in,
                 inputs.step,
                 top_k,
                 top_p,
                 temperature,
                 repetition_penalty,
                 no_repeat_ngram_size,
                 cum_log_probs_out.defined() && !segment.csr ?
                     std::optional<torch::Tensor>(cum_log_probs_out) : std::nullopt,
                 std::nullopt,  // output_log_probs
                 inputs.return_original_all_probs,
                 all_probs,
                 presence_penalty,
                 frequency_penalty,
                 do_sample,
                 generator,
                 greedy_sampling_buffer_ptr});
            if (greedy_output.success.defined()) {
                success.copy_(greedy_output.success);
            } else {
                success.fill_(true);
            }
            if (segment.csr) {
                auto tokens = token_ids_in.select(1, inputs.step).to(torch::kLong);
                auto in_range = tokens.ge(0).logical_and(tokens.lt(rq_scores.size(1)));
                auto selected = rq_scores.gather(1, tokens.clamp(0, rq_scores.size(1) - 1).unsqueeze(1)).squeeze(1);
                auto valid = in_range.logical_and(selected.isfinite()).logical_and(success);
                // Empty constraint rows terminate normally. A sampler failure on
                // a legal row is still reported through the ordinary error path.
                success.logical_or_(has_legal_token.squeeze(1).logical_not());
                auto valid_cpu = valid.cpu();
                const auto* valid_rows = valid_cpu.data_ptr<bool>();
                for (size_t row = 0; row < batch_size_in; ++row) {
                    effective_widths.emplace(from_batch_idx_out + row, valid_rows[row] ? 1 : 0);
                }
                if (cum_log_probs_out.defined()) {
                    cum_log_probs_out.add_(selected.to(cum_log_probs_out.device()));
                }
            }
            // execSampleGreedy updates token_ids_in in place. Mixed beam transitions use a
            // separate output tensor even when the aggregate input/output row counts match.
            if (requires_independent_output) {
                token_ids_out.copy_(token_ids_in);
            }
        } else {
            RTP_LLM_LOG_DEBUG("current_num_beams_in is %d", cur_num_beams_in);
            RTP_LLM_LOG_DEBUG("current_num_beams_out is %d", cur_num_beams_out);
            RTP_LLM_LOG_DEBUG("current_beam_batch is %d", beam_batch_size);
            RTP_LLM_CHECK_WITH_INFO((batch_size_in % cur_num_beams_in == 0),
                                    "sample_batch_size[%d] must devide by current_num_beams_in[%d]");

            const size_t vocab_size      = inputs.logits.size(1);
            const size_t max_seq_len_val = inputs.token_ids.size(1);

            auto beam_indices = all_beam_indices.narrow(0, from_batch_idx_out, batch_size_out);

            // Reshape for beam search: [batch, beams, ...]
            auto logits_reshaped =
                logits.reshape({(int64_t)beam_batch_size, (int64_t)cur_num_beams_in, (int64_t)vocab_size});
            auto token_ids_in_reshaped =
                token_ids_in.reshape({(int64_t)beam_batch_size, (int64_t)cur_num_beams_in, (int64_t)max_seq_len_val});
            auto input_lengths_reshaped = input_lengths.reshape({(int64_t)beam_batch_size, (int64_t)cur_num_beams_in});
            auto sequence_lengths_reshaped =
                sequence_lengths.reshape({(int64_t)beam_batch_size, (int64_t)cur_num_beams_in});
            auto cum_log_probs_in_reshaped =
                cum_log_probs_in.defined() ?
                    cum_log_probs_in.reshape({(int64_t)beam_batch_size, (int64_t)cur_num_beams_in}) :
                    torch::zeros({(int64_t)beam_batch_size, (int64_t)cur_num_beams_in});

            auto logits_t           = logits_reshaped.to(torch::kCUDA);
            auto token_ids_in_t     = token_ids_in_reshaped.to(torch::kCUDA);
            auto input_lengths_t    = input_lengths_reshaped.to(torch::kCUDA);
            auto sequence_lengths_t = sequence_lengths_reshaped.to(torch::kCUDA);
            auto cum_log_probs_in_t = cum_log_probs_in_reshaped.to(torch::kCUDA);

            if (segment.csr) {
                // Match TRT TopK's masked sentinel so invalid padding bypasses stable tie reordering.
                // CSR scores already contain log-probabilities; execSampleBeamSearch skips log_softmax.
                const char* fast_mask_env = std::getenv("RQ_FAST_MASK_TOPK");
                const float topk_mask = fast_mask_env && std::strcmp(fast_mask_env, "1") == 0 ?
                                            -std::numeric_limits<float>::infinity() : masked_beam_score;
                logits_t = rq_scores.masked_fill(rq_scores.isfinite().logical_not(), topk_mask)
                               .view_as(logits_t).to(torch::kCUDA);
            }
            auto output = execSampleBeamSearch({logits_t,
                                                token_ids_in_t,
                                                input_lengths_t,
                                                sequence_lengths_t,
                                                cum_log_probs_in_t,
                                                (size_t)cur_num_beams_out,
                                                segment.csr});

            auto token_ids_out_reshaped =
                token_ids_out.reshape({(int64_t)beam_batch_size, (int64_t)cur_num_beams_out, (int64_t)max_seq_len_val});
            auto cum_log_probs_out_reshaped =
                cum_log_probs_out.defined() ?
                    cum_log_probs_out.reshape({(int64_t)beam_batch_size, (int64_t)cur_num_beams_out}) :
                    torch::Tensor();

            token_ids_out_reshaped.copy_(output.token_ids);
            if (cum_log_probs_out_reshaped.defined()) {
                cum_log_probs_out_reshaped.copy_(output.cum_log_probs);
            }
            beam_indices.reshape({(int64_t)beam_batch_size, (int64_t)cur_num_beams_out}).copy_(output.beam_indices);

            if (segment.csr) {
                auto valid_cpu = output.cum_log_probs.gt(masked_beam_score / 2).cpu().contiguous();
                auto valid = valid_cpu.accessor<bool, 2>();
                for (size_t request = 0; request < beam_batch_size; ++request) {
                    std::vector<int64_t> keep;
                    for (int64_t beam = 0; beam < cur_num_beams_out; ++beam) {
                        if (valid[request][beam]) {
                            keep.push_back(beam);
                        }
                    }
                    const auto width = static_cast<int64_t>(keep.size());
                    const auto offset = from_batch_idx_out + request * cur_num_beams_out;
                    effective_widths.emplace(offset, width);
                    if (width > 0 && width < cur_num_beams_out) {
                        auto indices = torch::tensor(keep, torch::kLong);
                        token_ids_out_reshaped[request].narrow(0, 0, width).copy_(
                            output.token_ids[request].index_select(0, indices.to(output.token_ids.device())));
                        beam_indices.narrow(0, request * cur_num_beams_out, width).copy_(
                            output.beam_indices[request].index_select(0, indices.to(output.beam_indices.device())));
                        if (cum_log_probs_out_reshaped.defined()) {
                            cum_log_probs_out_reshaped[request].narrow(0, 0, width).copy_(
                                output.cum_log_probs[request].index_select(0, indices.to(output.cum_log_probs.device())));
                        }
                    }
                }
            }
            success.fill_(true);
        }
        from_batch_idx_out = to_batch_idx_out;
    }

    return SamplerOutput({std::move(all_token_ids_out),
                          std::move(all_cum_log_probs_out),
                          std::move(inputs.all_probs),
                          std::move(all_beam_indices),
                          std::move(all_success),
                          std::move(processor_errors),
                          std::move(effective_widths)});
}

std::vector<std::optional<ErrorInfo>> Sampler::preprocessLogits(const SamplerInputs& inputs) {
    if (inputs.logits_processor_states_ptr != nullptr) {
        return inputs.logits_processor_states_ptr->batchProcess(inputs);
    }
    return {};
}

}  // namespace rtp_llm
