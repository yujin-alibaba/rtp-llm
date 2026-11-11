#pragma once

#include "rtp_llm/cpp/models/logits_processor/BaseLogitsProcessor.h"

namespace rtp_llm {

// Each row of the trie stores sorted (token, child) edges in CSR order.
class RqCsrLogitsProcessor final: public BaseLogitsProcessor {
public:
    static void loadTokenBases(const std::string& checkpoint);
    static ErrorResult<BaseLogitsProcessorPtr> create(const GenerateConfig& config);

    std::optional<ErrorInfo> process(const SamplerInputs& inputs, size_t begin, size_t end) override;
    void updateMultiSeqStatus(const std::vector<int>&) override {}
    std::optional<ErrorInfo> updateStatus(const torch::Tensor&, int32_t) override {
        return std::nullopt;
    }

private:
    std::vector<size_t> row_offsets_;
    std::vector<int32_t> tokens_;
    std::vector<size_t> children_;
};

}  // namespace rtp_llm
