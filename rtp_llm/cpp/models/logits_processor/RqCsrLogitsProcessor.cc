#include "rtp_llm/cpp/models/logits_processor/RqCsrLogitsProcessor.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <fstream>
#include <iterator>
#include <limits>
#include <string_view>
#include <utility>

#include "absl/strings/escaping.h"
#include <rapidjson/document.h>

namespace rtp_llm {
namespace {

using Sid = std::array<int32_t, 3>;
Sid token_bases{};

bool parseSid(std::string_view text, Sid& sid) {
    for (size_t i = 0; i < sid.size(); ++i) {
        const auto separator = text.find('_');
        if ((i + 1 == sid.size()) != (separator == std::string_view::npos) || text.empty()) {
            return false;
        }
        const auto field = text.substr(0, separator);
        const auto parsed = std::from_chars(field.data(), field.data() + field.size(), sid[i]);
        if (parsed.ec != std::errc{} || parsed.ptr != field.data() + field.size() || sid[i] < 0) {
            return false;
        }
        if (separator != std::string_view::npos) {
            text.remove_prefix(separator + 1);
        }
    }
    return true;
}

uint64_t readLittleEndian(const char* bytes, size_t size) {
    uint64_t value = 0;
    for (size_t i = 0; i < size; ++i) {
        value |= uint64_t(static_cast<unsigned char>(bytes[i])) << (8 * i);
    }
    return value;
}

bool decodeSupply(const GenerateConfig& config, std::vector<Sid>& ids) {
    const bool flat = !config.extra_info.empty();
    if (!flat && config.ele_rq_ids_pb.empty()) {
        ids.resize(config.ele_rq_ids.size());
        for (size_t i = 0; i < ids.size(); ++i) {
            if (!parseSid(config.ele_rq_ids[i], ids[i])) {
                return false;
            }
        }
        return true;
    }

    std::string bytes;
    if (!absl::Base64Unescape(flat ? config.extra_info : config.ele_rq_ids_pb, &bytes)) {
        return false;
    }
    const size_t header = flat ? 0 : 4;
    const size_t stride = flat ? 6 : 8;
    if (bytes.size() < header || (bytes.size() - header) % stride != 0) {
        return false;
    }
    const size_t count = (bytes.size() - header) / stride;
    if (!flat && readLittleEndian(bytes.data(), header) != count) {
        return false;
    }
    ids.resize(count);
    for (size_t i = 0; i < count; ++i) {
        const char* row = bytes.data() + header + stride * i;
        if (flat) {
            for (size_t layer = 0; layer < 3; ++layer) {
                const auto token = uint64_t(token_bases[layer]) + readLittleEndian(row + 2 * layer, 2);
                if (token > std::numeric_limits<int32_t>::max()) {
                    return false;
                }
                ids[i][layer] = static_cast<int32_t>(token);
            }
        } else {
            const uint64_t packed = readLittleEndian(row, 8);
            if (packed >> 54) {
                return false;
            }
            ids[i] = {int32_t(packed >> 36), int32_t((packed >> 18) & 0x3ffff), int32_t(packed & 0x3ffff)};
        }
    }
    return true;
}

}  // namespace

void RqCsrLogitsProcessor::loadTokenBases(const std::string& checkpoint) {
    token_bases = {};
    std::ifstream file(checkpoint + "/tokenizer_config.json");
    if (!file) {
        file.open(checkpoint + "/tokenizer/tokenizer_config.json");
    }
    const std::string text((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    rapidjson::Document config;
    config.Parse(text.data(), text.size());
    if (config.HasParseError() || !config.IsObject() || !config.HasMember("added_tokens_decoder")
        || !config["added_tokens_decoder"].IsObject()) {
        return;
    }
    const auto decoder = config["added_tokens_decoder"].GetObject();
    Sid bases{};
    std::array<bool, 3> found{};
    for (const auto& member : decoder) {
        if (!member.value.IsObject() || !member.value.HasMember("content") || !member.value["content"].IsString()) {
            continue;
        }
        const auto& content = member.value["content"];
        const std::string_view token(content.GetString(), content.GetStringLength());
        const std::array<std::string_view, 3> names{"<shop_0_0>", "<shop_1_0>", "<shop_2_0>"};
        const auto name = std::find(names.begin(), names.end(), token);
        if (name == names.end()) {
            continue;
        }
        const auto layer = std::distance(names.begin(), name);
        const auto* begin = member.name.GetString();
        const auto* end = begin + member.name.GetStringLength();
        const auto parsed = std::from_chars(begin, end, bases[layer]);
        found[layer] = parsed.ec == std::errc{} && parsed.ptr == end && bases[layer] >= 0;
    }
    if (std::all_of(found.begin(), found.end(), [](bool value) { return value; })) {
        token_bases = bases;
    }
}

ErrorResult<BaseLogitsProcessorPtr> RqCsrLogitsProcessor::create(const GenerateConfig& config) {
    std::vector<Sid> ids;
    if (!decodeSupply(config, ids)) {
        return ErrorInfo(ErrorCode::INVALID_PARAMS, "invalid RQ supply encoding");
    }
    std::sort(ids.begin(), ids.end());
    struct Prefix {
        size_t begin;
        size_t end;
        size_t depth;
    };
    std::vector<Prefix> rows{{0, ids.size(), 0}};
    auto processor = std::make_shared<RqCsrLogitsProcessor>();
    for (size_t row = 0; row < rows.size(); ++row) {
        const auto [begin, end, depth] = rows[row];
        processor->row_offsets_.push_back(processor->tokens_.size());
        if (depth == Sid{}.size()) {
            continue;
        }
        for (size_t first = begin; first < end;) {
            const auto token = ids[first][depth];
            size_t next = first + 1;
            while (next < end && ids[next][depth] == token) {
                ++next;
            }
            processor->tokens_.push_back(token);
            processor->children_.push_back(rows.size());
            rows.push_back({first, next, depth + 1});
            first = next;
        }
    }
    processor->row_offsets_.push_back(processor->tokens_.size());
    return BaseLogitsProcessorPtr(std::move(processor));
}

std::optional<ErrorInfo> RqCsrLogitsProcessor::process(const SamplerInputs& inputs, size_t begin, size_t end) {
    auto logits = inputs.logits.narrow(0, begin, end - begin);
    const auto vocab = logits.size(1);
    const auto history = inputs.token_ids.accessor<int32_t, 2>();
    const auto* prompt = inputs.input_lengths.data_ptr<int32_t>();
    const auto* sequence = inputs.sequence_lengths.data_ptr<int32_t>();
    std::vector<int64_t> allowed;
    for (size_t row = begin; row < end; ++row) {
        size_t node = 0;
        for (int32_t pos = prompt[row]; pos < sequence[row]; ++pos) {
            const auto first = tokens_.begin() + row_offsets_[node];
            const auto last = tokens_.begin() + row_offsets_[node + 1];
            const auto edge = std::lower_bound(first, last, history[row][pos]);
            if (edge == last || *edge != history[row][pos]) {
                node = row_offsets_.size() - 1;
                break;
            }
            node = children_[std::distance(tokens_.begin(), edge)];
        }
        if (node == row_offsets_.size() - 1) {
            continue;
        }
        for (size_t edge = row_offsets_[node]; edge < row_offsets_[node + 1]; ++edge) {
            const auto token = tokens_[edge];
            if (token < vocab) {
                allowed.push_back(int64_t(row - begin) * vocab + token);
            }
        }
    }

    auto mask = torch::ones({logits.numel()}, logits.options().dtype(torch::kBool));
    if (!allowed.empty()) {
        auto indices = torch::tensor(allowed, torch::kInt64).to(logits.device());
        mask.index_fill_(0, indices, false);
    }
    logits.masked_fill_(mask.view_as(logits), -std::numeric_limits<float>::infinity());
    return std::nullopt;
}

}  // namespace rtp_llm
