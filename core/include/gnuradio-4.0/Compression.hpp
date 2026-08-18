#ifndef GNURADIO_COMPRESSION_HPP
#define GNURADIO_COMPRESSION_HPP

#include <gnuradio-4.0/CRC.hpp>
#include <gnuradio-4.0/meta/utils.hpp>

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <limits>
#include <new>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace gr::compression {

/**
 * @brief Constant-evaluated gzip resources and allocation-free DEFLATE decoding.
 *
 * The format implementation follows RFC 1950 (zlib), RFC 1951 (DEFLATE), and
 * RFC 1952 (gzip):
 * - https://www.rfc-editor.org/rfc/rfc1950
 * - https://www.rfc-editor.org/rfc/rfc1951
 * - https://www.rfc-editor.org/rfc/rfc1952
 *
 * Text resources:
 * ```cpp
 * using namespace gr::compression::literals;
 * static constexpr auto help = "embedded help text"_gzip;
 * static constexpr auto compactHelp = gr::compression::makeCompressedText<
 *     "embedded help text", gr::compression::CompressionLevel::best>();
 * std::string_view view = help;
 * ```
 *
 * Binary resources:
 * ```cpp
 * inline constexpr std::array source{std::byte{0x01}, std::byte{0x02}};
 * inline constexpr auto packed = gr::compression::gzip<source, gr::compression::CompressionLevel::best>();
 * ```
 * `fast`, `balanced`, and `best` increase match-search effort; `best` also uses
 * lazy matching. The `_gzip` literal and APIs without a level use `balanced`.
 */

enum class Format : std::uint8_t { rawDeflate, zlib, gzip };

enum class CompressionLevel : std::uint8_t { fast, balanced, best };

enum class Error : std::uint8_t { truncatedInput, outputTooSmall, invalidHeader, unsupportedCompressionMethod, reservedFlags, presetDictionaryRequired, invalidBlockType, invalidStoredBlockLength, invalidHuffmanTree, invalidCode, invalidLength, invalidDistance, invalidBackReference, checksumMismatch, sizeMismatch, trailingData, integerOverflow, allocationFailed };

[[nodiscard]] constexpr std::uint32_t adler32(std::span<const std::byte> input) noexcept {
    constexpr std::uint32_t kModAdler = 65521U;
    std::uint32_t           a         = 1U;
    std::uint32_t           b         = 0U;
    for (const std::byte byte : input) {
        a = (a + std::to_integer<std::uint8_t>(byte)) % kModAdler;
        b = (b + a) % kModAdler;
    }
    return (b << 16U) | a;
}

namespace detail {

inline constexpr std::array<std::uint16_t, 29> kLengthBase{3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258};
inline constexpr std::array<std::uint8_t, 29>  kLengthExtra{0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0};
inline constexpr std::array<std::uint16_t, 30> kDistanceBase{1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577};
inline constexpr std::array<std::uint8_t, 30>  kDistanceExtra{0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13};
inline constexpr std::array<std::uint8_t, 19>  kCodeLengthOrder{16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15};

[[nodiscard]] constexpr std::uint16_t reverseBits(std::uint16_t value, unsigned count) noexcept {
    std::uint16_t result = 0;
    for (unsigned i = 0; i < count; ++i) {
        result = static_cast<std::uint16_t>((static_cast<unsigned>(result) << 1U) | (static_cast<unsigned>(value) & 1U));
        value >>= 1U;
    }
    return result;
}

struct EncodedSymbol {
    unsigned symbol{};
    unsigned extraValue{};
    unsigned extraBits{};
};

[[nodiscard]] constexpr EncodedSymbol encodeLength(std::size_t length) noexcept {
    std::size_t index = 0;
    while (index + 1UZ < kLengthBase.size() && kLengthBase[index + 1UZ] <= length) {
        ++index;
    }
    return {.symbol = static_cast<unsigned>(257UZ + index), .extraValue = static_cast<unsigned>(length - kLengthBase[index]), .extraBits = kLengthExtra[index]};
}

[[nodiscard]] constexpr EncodedSymbol encodeDistance(std::size_t distance) noexcept {
    std::size_t index = 0;
    while (index + 1UZ < kDistanceBase.size() && kDistanceBase[index + 1UZ] <= distance) {
        ++index;
    }
    return {.symbol = static_cast<unsigned>(index), .extraValue = static_cast<unsigned>(distance - kDistanceBase[index]), .extraBits = kDistanceExtra[index]};
}

template<std::size_t N>
struct ArraySink {
    std::array<std::byte, N>& out;
    std::size_t               pos{};

    constexpr void put(std::byte byte) noexcept { out[pos++] = byte; }
};

template<typename Sink>
struct BitWriter {
    Sink&         sink;
    std::uint64_t bits{};
    unsigned      count{};

    constexpr void writeBits(std::uint32_t value, unsigned width) noexcept {
        bits |= static_cast<std::uint64_t>(value) << count;
        count += width;
        while (count >= 8U) {
            sink.put(static_cast<std::byte>(bits & 0xffU));
            bits >>= 8U;
            count -= 8U;
        }
    }

    constexpr void alignByte() noexcept {
        if (count != 0U) {
            sink.put(static_cast<std::byte>(bits & 0xffU));
            bits  = 0;
            count = 0;
        }
    }
};

template<typename Sink>
constexpr void putLittle16(Sink& sink, std::uint16_t value) noexcept {
    sink.put(static_cast<std::byte>(value & 0xffU));
    sink.put(static_cast<std::byte>(value >> 8U));
}

template<typename Sink>
constexpr void putLittle32(Sink& sink, std::uint32_t value) noexcept {
    for (unsigned shift = 0; shift < 32U; shift += 8U) {
        sink.put(static_cast<std::byte>((value >> shift) & 0xffU));
    }
}

struct HuffmanCode {
    std::uint16_t bits{};
    std::uint8_t  width{};
};

[[nodiscard]] constexpr HuffmanCode fixedLiteralCode(unsigned symbol) noexcept {
    if (symbol <= 143U) {
        return {.bits = reverseBits(static_cast<std::uint16_t>(0x30U + symbol), 8), .width = 8};
    }
    if (symbol <= 255U) {
        return {.bits = reverseBits(static_cast<std::uint16_t>(0x190U + symbol - 144U), 9), .width = 9};
    }
    if (symbol <= 279U) {
        return {.bits = reverseBits(static_cast<std::uint16_t>(symbol - 256U), 7), .width = 7};
    }
    return {.bits = reverseBits(static_cast<std::uint16_t>(0xc0U + symbol - 280U), 8), .width = 8};
}

[[nodiscard]] constexpr HuffmanCode fixedDistanceCode(unsigned symbol) noexcept { return {.bits = reverseBits(static_cast<std::uint16_t>(symbol), 5), .width = 5}; }

struct Token {
    std::uint16_t value{};
    std::uint16_t distance{};

    [[nodiscard]] constexpr bool isLiteral() const noexcept { return distance == 0U; }
};

template<std::size_t N>
struct TokenBuffer {
    std::array<Token, N == 0UZ ? 1UZ : N> value{};
    std::size_t                           size{};

    constexpr void push(Token token) noexcept { value[size++] = token; }
};

struct Match {
    std::size_t length{};
    std::size_t distance{};
};

template<CompressionLevel Level>
struct EncodePolicy {
    static_assert(Level == CompressionLevel::fast || Level == CompressionLevel::balanced || Level == CompressionLevel::best);

    static constexpr std::size_t kMaxCandidates = Level == CompressionLevel::fast ? 4UZ : Level == CompressionLevel::balanced ? 16UZ : 128UZ;
    static constexpr bool        kLazyMatching  = Level == CompressionLevel::best;
};

template<auto Data, CompressionLevel Level>
struct Matcher {
    static constexpr std::size_t kWindowSize   = 32768UZ;
    static constexpr std::size_t kHashSize     = 65536UZ;
    static constexpr std::size_t kPreviousSize = Data.size() == 0UZ ? 1UZ : std::min(Data.size(), kWindowSize);

    std::array<std::size_t, kHashSize>     head{};
    std::array<std::size_t, kPreviousSize> previous{};

    [[nodiscard]] static constexpr std::size_t hash(std::size_t pos) noexcept {
        const auto a     = std::to_integer<std::uint32_t>(Data[pos]);
        const auto b     = std::to_integer<std::uint32_t>(Data[pos + 1UZ]);
        const auto c     = std::to_integer<std::uint32_t>(Data[pos + 2UZ]);
        const auto value = (a << 16U) | (b << 8U) | c;
        return (value * 2654435761U) >> 16U;
    }

    constexpr void insert(std::size_t pos) noexcept {
        if (pos + 2UZ >= Data.size()) {
            return;
        }
        const auto key                = hash(pos);
        previous[pos % kPreviousSize] = head[key];
        head[key]                     = pos + 1UZ;
    }

    [[nodiscard]] constexpr Match find(std::size_t pos) const noexcept {
        if (pos + 2UZ >= Data.size()) {
            return {};
        }

        Match       best{};
        std::size_t candidate = head[hash(pos)];
        std::size_t checked   = 0;
        while (candidate != 0UZ && candidate - 1UZ < pos && pos - (candidate - 1UZ) <= kWindowSize && checked < EncodePolicy<Level>::kMaxCandidates) {
            --candidate;
            ++checked;
            std::size_t length = 0;
            while (length < 258UZ && pos + length < Data.size() && Data[candidate + length] == Data[pos + length]) {
                ++length;
            }
            if (length >= 3UZ && length > best.length) {
                best = {.length = length, .distance = pos - candidate};
                if (length == 258UZ) {
                    break;
                }
            }
            candidate = previous[candidate % kPreviousSize];
        }
        return best;
    }
};

template<auto Data, CompressionLevel Level>
[[nodiscard]] consteval auto tokenize() {
    TokenBuffer<Data.size()> tokens{};
    Matcher<Data, Level>     matcher{};
    std::size_t              pos = 0;
    while (pos < Data.size()) {
        const auto  match       = matcher.find(pos);
        std::size_t firstInsert = 0;
        if constexpr (EncodePolicy<Level>::kLazyMatching) {
            if (match.length >= 3UZ) {
                matcher.insert(pos);
                firstInsert = 1;
                if (matcher.find(pos + 1UZ).length > match.length) {
                    tokens.push({.value = std::to_integer<std::uint8_t>(Data[pos]), .distance = 0});
                    ++pos;
                    continue;
                }
            }
        }
        if (match.length >= 3UZ) {
            tokens.push({.value = static_cast<std::uint16_t>(match.length), .distance = static_cast<std::uint16_t>(match.distance)});
            for (std::size_t i = firstInsert; i < match.length; ++i) {
                matcher.insert(pos + i);
            }
            pos += match.length;
        } else {
            tokens.push({.value = std::to_integer<std::uint8_t>(Data[pos]), .distance = 0});
            matcher.insert(pos);
            ++pos;
        }
    }
    return tokens;
}

template<std::size_t N>
struct CanonicalCodes {
    std::array<std::uint8_t, N>  lengths{};
    std::array<std::uint16_t, N> bits{};
};

template<std::size_t N>
[[nodiscard]] constexpr CanonicalCodes<N> makeCanonicalCodes(std::array<std::uint8_t, N> lengths) noexcept {
    CanonicalCodes<N>             result{.lengths = lengths};
    std::array<std::uint16_t, 16> counts{};
    std::array<std::uint16_t, 16> next{};
    for (const auto length : lengths) {
        if (length != 0U) {
            ++counts[length];
        }
    }
    std::uint16_t code = 0;
    for (std::size_t width = 1; width < counts.size(); ++width) {
        code        = static_cast<std::uint16_t>((code + counts[width - 1UZ]) << 1U);
        next[width] = code;
    }
    for (std::size_t symbol = 0; symbol < N; ++symbol) {
        const auto width = lengths[symbol];
        if (width != 0U) {
            result.bits[symbol] = reverseBits(next[width]++, width);
        }
    }
    return result;
}

template<std::size_t N, unsigned MaxBits>
[[nodiscard]] constexpr std::array<std::uint8_t, N> makeCodeLengths(std::array<std::uint32_t, N> frequencies) noexcept {
    constexpr std::size_t kNone  = std::numeric_limits<std::size_t>::max();
    std::size_t           active = 0;
    for (const auto frequency : frequencies) {
        active += frequency != 0U ? 1UZ : 0UZ;
    }
    for (std::size_t symbol = 0; active < 2UZ && symbol < N; ++symbol) {
        if (frequencies[symbol] == 0U) {
            frequencies[symbol] = 1U;
            ++active;
        }
    }

    std::array<std::uint64_t, 2UZ * N> nodeFrequency{};
    std::array<std::size_t, 2UZ * N>   parent{};
    parent.fill(kNone);
    for (std::size_t i = 0; i < N; ++i) {
        nodeFrequency[i] = frequencies[i];
    }

    std::array<std::size_t, 2UZ * N> heap{};
    std::size_t                      heapSize = 0;
    const auto                       less     = [&nodeFrequency](std::size_t left, std::size_t right) constexpr { return nodeFrequency[left] < nodeFrequency[right] || (nodeFrequency[left] == nodeFrequency[right] && left < right); };
    const auto                       push     = [&heap, &heapSize, &less](std::size_t node) constexpr {
        std::size_t index = heapSize++;
        while (index > 0UZ) {
            const auto parentIndex = (index - 1UZ) / 2UZ;
            if (!less(node, heap[parentIndex])) {
                break;
            }
            heap[index] = heap[parentIndex];
            index       = parentIndex;
        }
        heap[index] = node;
    };
    const auto pop = [&heap, &heapSize, &less]() constexpr {
        const auto  result = heap[0];
        const auto  last   = heap[--heapSize];
        std::size_t index  = 0;
        while (index * 2UZ + 1UZ < heapSize) {
            auto child = index * 2UZ + 1UZ;
            if (child + 1UZ < heapSize && less(heap[child + 1UZ], heap[child])) {
                ++child;
            }
            if (!less(heap[child], last)) {
                break;
            }
            heap[index] = heap[child];
            index       = child;
        }
        if (heapSize != 0UZ) {
            heap[index] = last;
        }
        return result;
    };

    for (std::size_t symbol = 0; symbol < N; ++symbol) {
        if (frequencies[symbol] != 0U) {
            push(symbol);
        }
    }
    std::size_t nodeCount = N;
    while (heapSize > 1UZ) {
        const auto first         = pop();
        const auto second        = pop();
        parent[first]            = nodeCount;
        parent[second]           = nodeCount;
        nodeFrequency[nodeCount] = nodeFrequency[first] + nodeFrequency[second];
        push(nodeCount++);
    }

    std::array<std::uint8_t, N>         lengths{};
    std::array<unsigned, MaxBits + 1UZ> lengthCounts{};
    std::array<unsigned, 2UZ * N>       depths{};
    for (std::size_t node = nodeCount; node-- > 0UZ;) {
        if (parent[node] != kNone) {
            depths[node] = depths[parent[node]] + 1U;
        }
    }
    for (std::size_t symbol = 0; symbol < N; ++symbol) {
        if (frequencies[symbol] == 0U) {
            continue;
        }
        unsigned length = depths[symbol];
        if (length > MaxBits) {
            length = MaxBits;
        }
        lengths[symbol] = static_cast<std::uint8_t>(length);
        ++lengthCounts[length];
    }

    std::size_t usedCodeSpace = 0;
    for (unsigned width = 1; width <= MaxBits; ++width) {
        usedCodeSpace += static_cast<std::size_t>(lengthCounts[width]) << (MaxBits - width);
    }
    const auto availableCodeSpace = std::size_t{1} << MaxBits;
    while (usedCodeSpace > availableCodeSpace) {
        unsigned width = MaxBits - 1U;
        while (width > 0U && lengthCounts[width] == 0U) {
            --width;
        }
        --lengthCounts[width];
        lengthCounts[width + 1U] += 2U;
        --lengthCounts[MaxBits];
        --usedCodeSpace;
    }

    std::array<std::size_t, N> ordered{};
    std::size_t                orderedSize = 0;
    heapSize                               = 0;
    for (std::size_t symbol = 0; symbol < N; ++symbol) {
        if (frequencies[symbol] != 0U) {
            push(symbol);
        }
    }
    while (heapSize != 0UZ) {
        ordered[orderedSize++] = pop();
    }

    lengths.fill(0U);
    std::size_t orderedIndex = 0;
    for (unsigned width = MaxBits; width > 0U; --width) {
        for (unsigned i = 0; i < lengthCounts[width]; ++i) {
            lengths[ordered[orderedIndex++]] = static_cast<std::uint8_t>(width);
        }
    }
    return lengths;
}

struct CodeLengthToken {
    std::uint8_t symbol{};
    std::uint8_t extraValue{};
    std::uint8_t extraBits{};
};

struct CodeLengthTokens {
    std::array<CodeLengthToken, 632> value{};
    std::size_t                      size{};

    constexpr void push(std::uint8_t symbol, std::uint8_t extraValue = 0, std::uint8_t extraBits = 0) noexcept { value[size++] = {.symbol = symbol, .extraValue = extraValue, .extraBits = extraBits}; }
};

[[nodiscard]] constexpr CodeLengthTokens encodeCodeLengths(std::span<const std::uint8_t> lengths) noexcept {
    CodeLengthTokens result{};
    std::size_t      pos = 0;
    while (pos < lengths.size()) {
        const auto  length = lengths[pos];
        std::size_t run    = 1;
        while (pos + run < lengths.size() && lengths[pos + run] == length) {
            ++run;
        }
        const auto originalRun = run;
        if (length == 0U) {
            while (run >= 11UZ) {
                const auto count = std::min(run, 138UZ);
                result.push(18, static_cast<std::uint8_t>(count - 11UZ), 7);
                run -= count;
            }
            if (run >= 3UZ) {
                const auto count = std::min(run, 10UZ);
                result.push(17, static_cast<std::uint8_t>(count - 3UZ), 3);
                run -= count;
            }
            while (run-- > 0UZ) {
                result.push(0);
            }
        } else {
            result.push(length);
            --run;
            while (run >= 3UZ) {
                const auto count = std::min(run, 6UZ);
                result.push(16, static_cast<std::uint8_t>(count - 3UZ), 2);
                run -= count;
            }
            while (run-- > 0UZ) {
                result.push(length);
            }
        }
        pos += originalRun;
    }
    return result;
}

struct DynamicCoding {
    CanonicalCodes<286> literals{};
    CanonicalCodes<30>  distances{};
    CanonicalCodes<19>  codeLengths{};
    CodeLengthTokens    encodedLengths{};
    std::uint16_t       literalCount{};
    std::uint8_t        distanceCount{};
    std::uint8_t        codeLengthCount{};
};

template<std::size_t N>
[[nodiscard]] constexpr DynamicCoding makeDynamicCoding(const TokenBuffer<N>& tokens) noexcept {
    std::array<std::uint32_t, 286> literalFrequencies{};
    std::array<std::uint32_t, 30>  distanceFrequencies{};
    for (std::size_t i = 0; i < tokens.size; ++i) {
        const auto token = tokens.value[i];
        if (token.isLiteral()) {
            ++literalFrequencies[token.value];
        } else {
            ++literalFrequencies[encodeLength(token.value).symbol];
            ++distanceFrequencies[encodeDistance(token.distance).symbol];
        }
    }
    ++literalFrequencies[256];

    DynamicCoding result{};
    result.literals  = makeCanonicalCodes(makeCodeLengths<286, 15>(literalFrequencies));
    result.distances = makeCanonicalCodes(makeCodeLengths<30, 15>(distanceFrequencies));

    std::size_t literalCount = 286;
    while (literalCount > 257UZ && result.literals.lengths[literalCount - 1UZ] == 0U) {
        --literalCount;
    }
    std::size_t distanceCount = 30;
    while (distanceCount > 1UZ && result.distances.lengths[distanceCount - 1UZ] == 0U) {
        --distanceCount;
    }
    result.literalCount  = static_cast<std::uint16_t>(literalCount);
    result.distanceCount = static_cast<std::uint8_t>(distanceCount);

    std::array<std::uint8_t, 316> combined{};
    for (std::size_t i = 0; i < literalCount; ++i) {
        combined[i] = result.literals.lengths[i];
    }
    for (std::size_t i = 0; i < distanceCount; ++i) {
        combined[literalCount + i] = result.distances.lengths[i];
    }
    result.encodedLengths = encodeCodeLengths(std::span<const std::uint8_t>(combined.data(), literalCount + distanceCount));

    std::array<std::uint32_t, 19> codeLengthFrequencies{};
    for (std::size_t i = 0; i < result.encodedLengths.size; ++i) {
        ++codeLengthFrequencies[result.encodedLengths.value[i].symbol];
    }
    result.codeLengths = makeCanonicalCodes(makeCodeLengths<19, 7>(codeLengthFrequencies));

    std::size_t codeLengthCount = 19;
    while (codeLengthCount > 4UZ && result.codeLengths.lengths[kCodeLengthOrder[codeLengthCount - 1UZ]] == 0U) {
        --codeLengthCount;
    }
    result.codeLengthCount = static_cast<std::uint8_t>(codeLengthCount);
    return result;
}

template<typename Writer, std::size_t N>
constexpr void writeCode(Writer& writer, const CanonicalCodes<N>& codes, unsigned symbol) noexcept {
    writer.writeBits(codes.bits[symbol], codes.lengths[symbol]);
}

template<typename Writer, std::size_t N>
constexpr void emitTokensFixed(Writer& writer, const TokenBuffer<N>& tokens) noexcept {
    for (std::size_t i = 0; i < tokens.size; ++i) {
        const auto token = tokens.value[i];
        if (token.isLiteral()) {
            const auto code = fixedLiteralCode(token.value);
            writer.writeBits(code.bits, code.width);
            continue;
        }
        const auto length     = encodeLength(token.value);
        const auto lengthCode = fixedLiteralCode(length.symbol);
        writer.writeBits(lengthCode.bits, lengthCode.width);
        writer.writeBits(length.extraValue, length.extraBits);
        const auto distance     = encodeDistance(token.distance);
        const auto distanceCode = fixedDistanceCode(distance.symbol);
        writer.writeBits(distanceCode.bits, distanceCode.width);
        writer.writeBits(distance.extraValue, distance.extraBits);
    }
    const auto end = fixedLiteralCode(256);
    writer.writeBits(end.bits, end.width);
}

template<typename Writer, std::size_t N>
constexpr void emitTokensDynamic(Writer& writer, const TokenBuffer<N>& tokens, const DynamicCoding& coding) noexcept {
    for (std::size_t i = 0; i < tokens.size; ++i) {
        const auto token = tokens.value[i];
        if (token.isLiteral()) {
            writeCode(writer, coding.literals, token.value);
            continue;
        }
        const auto length = encodeLength(token.value);
        writeCode(writer, coding.literals, length.symbol);
        writer.writeBits(length.extraValue, length.extraBits);
        const auto distance = encodeDistance(token.distance);
        writeCode(writer, coding.distances, distance.symbol);
        writer.writeBits(distance.extraValue, distance.extraBits);
    }
    writeCode(writer, coding.literals, 256);
}

enum class Encoding : std::uint8_t { stored, fixed, dynamic };

template<auto Data, typename Sink>
constexpr void emitStoredDeflate(Sink& sink) noexcept {
    BitWriter writer{sink};
    if constexpr (Data.size() == 0UZ) {
        writer.writeBits(1U, 1);
        writer.writeBits(0U, 2);
        writer.alignByte();
        putLittle16(sink, 0);
        putLittle16(sink, 0xffffU);
        return;
    }

    std::size_t pos = 0;
    while (pos < Data.size()) {
        const auto size  = std::min<std::size_t>(Data.size() - pos, 65535UZ);
        const auto final = pos + size == Data.size();
        writer.writeBits(final ? 1U : 0U, 1);
        writer.writeBits(0U, 2);
        writer.alignByte();
        putLittle16(sink, static_cast<std::uint16_t>(size));
        putLittle16(sink, static_cast<std::uint16_t>(~static_cast<std::uint16_t>(size)));
        for (std::size_t i = 0; i < size; ++i) {
            sink.put(Data[pos + i]);
        }
        pos += size;
    }
}

template<typename Sink, std::size_t N>
constexpr void emitFixedDeflate(Sink& sink, const TokenBuffer<N>& tokens) noexcept {
    BitWriter writer{sink};
    writer.writeBits(1U, 1);
    writer.writeBits(1U, 2);
    emitTokensFixed(writer, tokens);
    writer.alignByte();
}

template<typename Sink, std::size_t N>
constexpr void emitDynamicDeflate(Sink& sink, const TokenBuffer<N>& tokens, const DynamicCoding& coding) noexcept {
    BitWriter writer{sink};
    writer.writeBits(1U, 1);
    writer.writeBits(2U, 2);
    writer.writeBits(coding.literalCount - 257U, 5);
    writer.writeBits(coding.distanceCount - 1U, 5);
    writer.writeBits(coding.codeLengthCount - 4U, 4);
    for (std::size_t i = 0; i < coding.codeLengthCount; ++i) {
        writer.writeBits(coding.codeLengths.lengths[kCodeLengthOrder[i]], 3);
    }
    for (std::size_t i = 0; i < coding.encodedLengths.size; ++i) {
        const auto token = coding.encodedLengths.value[i];
        writeCode(writer, coding.codeLengths, token.symbol);
        writer.writeBits(token.extraValue, token.extraBits);
    }
    emitTokensDynamic(writer, tokens, coding);
    writer.alignByte();
}

template<auto Data, std::size_t N>
[[nodiscard]] constexpr std::size_t deflateSize(Encoding encoding, const TokenBuffer<N>& tokens, const DynamicCoding& coding) noexcept {
    switch (encoding) {
    case Encoding::stored: {
        constexpr auto blocks = Data.empty() ? 1UZ : (Data.size() + 65534UZ) / 65535UZ;
        return Data.size() + blocks * 5UZ;
    }
    case Encoding::fixed: {
        std::size_t bits = 10UZ;
        for (std::size_t i = 0; i < tokens.size; ++i) {
            const auto token = tokens.value[i];
            if (token.isLiteral()) {
                bits += token.value <= 143U ? 8UZ : 9UZ;
            } else {
                const auto length   = encodeLength(token.value);
                const auto distance = encodeDistance(token.distance);
                bits += (length.symbol <= 279U ? 7UZ : 8UZ) + length.extraBits + 5UZ + distance.extraBits;
            }
        }
        return (bits + 7UZ) / 8UZ;
    }
    case Encoding::dynamic: {
        std::size_t bits = 17UZ + 3UZ * coding.codeLengthCount + coding.literals.lengths[256];
        for (std::size_t i = 0; i < coding.encodedLengths.size; ++i) {
            const auto token = coding.encodedLengths.value[i];
            bits += static_cast<std::size_t>(coding.codeLengths.lengths[token.symbol]) + token.extraBits;
        }
        for (std::size_t i = 0; i < tokens.size; ++i) {
            const auto token = tokens.value[i];
            if (token.isLiteral()) {
                bits += coding.literals.lengths[token.value];
            } else {
                const auto length   = encodeLength(token.value);
                const auto distance = encodeDistance(token.distance);
                bits += coding.literals.lengths[length.symbol] + length.extraBits + coding.distances.lengths[distance.symbol] + distance.extraBits;
            }
        }
        return (bits + 7UZ) / 8UZ;
    }
    }
    return 0;
}

template<auto Data, std::size_t N>
[[nodiscard]] constexpr Encoding selectEncoding(const TokenBuffer<N>& tokens, const DynamicCoding& coding) noexcept {
    Encoding    selected = Encoding::fixed;
    std::size_t size     = deflateSize<Data>(selected, tokens, coding);
    if (const auto dynamicSize = deflateSize<Data>(Encoding::dynamic, tokens, coding); dynamicSize < size) {
        selected = Encoding::dynamic;
        size     = dynamicSize;
    }
    if (const auto storedSize = deflateSize<Data>(Encoding::stored, tokens, coding); storedSize < size) {
        selected = Encoding::stored;
    }
    return selected;
}

template<auto Data, typename Sink, std::size_t N>
constexpr void emitDeflate(Sink& sink, Encoding encoding, const TokenBuffer<N>& tokens, const DynamicCoding& coding) noexcept {
    switch (encoding) {
    case Encoding::stored: emitStoredDeflate<Data>(sink); break;
    case Encoding::fixed: emitFixedDeflate(sink, tokens); break;
    case Encoding::dynamic: emitDynamicDeflate(sink, tokens, coding); break;
    }
}

template<gr::meta::fixed_string S>
[[nodiscard]] consteval auto staticBytesFromString() {
    std::array<std::byte, S.size()> bytes{};
    for (std::size_t i = 0; i < S.size(); ++i) {
        bytes[i] = static_cast<std::byte>(static_cast<unsigned char>(S[i]));
    }
    return bytes;
}

struct BitReader {
    std::span<const std::byte> input;
    std::size_t                pos{};
    std::uint64_t              bits{};
    unsigned                   count{};

    [[nodiscard]] constexpr std::expected<std::uint32_t, Error> readBits(unsigned width) noexcept {
        if (width > 32U) {
            return std::unexpected(Error::invalidLength);
        }
        while (count < width) {
            if (pos == input.size()) {
                return std::unexpected(Error::truncatedInput);
            }
            bits |= static_cast<std::uint64_t>(std::to_integer<std::uint8_t>(input[pos++])) << count;
            count += 8U;
        }
        const auto mask   = width == 32U ? 0xffff'ffffULL : ((std::uint64_t{1} << width) - 1U);
        const auto result = static_cast<std::uint32_t>(bits & mask);
        bits >>= width;
        count -= width;
        return result;
    }

    constexpr void alignByte() noexcept {
        const auto discard = count & 7U;
        bits >>= discard;
        count -= discard;
    }
};

template<std::size_t N, unsigned MaxBits>
struct HuffmanDecoder {
    std::array<std::uint16_t, MaxBits + 1UZ> counts{};
    std::array<std::uint16_t, N>             symbols{};
    std::uint8_t                             maxWidth{};
};

template<std::size_t N, unsigned MaxBits>
[[nodiscard]] constexpr std::expected<HuffmanDecoder<N, MaxBits>, Error> makeDecoder(const std::array<std::uint8_t, N>& lengths, std::size_t symbolCount = N) noexcept {
    HuffmanDecoder<N, MaxBits> result{};
    std::size_t                used = 0;
    for (std::size_t symbol = 0; symbol < symbolCount; ++symbol) {
        const auto length = lengths[symbol];
        if (length > MaxBits) {
            return std::unexpected(Error::invalidHuffmanTree);
        }
        if (length != 0U) {
            ++result.counts[length];
            result.maxWidth = std::max(result.maxWidth, length);
            ++used;
        }
    }
    if (used == 0UZ) {
        return std::unexpected(Error::invalidHuffmanTree);
    }

    int remaining = 1;
    for (unsigned width = 1; width <= MaxBits; ++width) {
        remaining = (remaining << 1) - result.counts[width];
        if (remaining < 0) {
            return std::unexpected(Error::invalidHuffmanTree);
        }
    }

    std::size_t index = 0;
    for (unsigned width = 1; width <= MaxBits; ++width) {
        for (std::size_t symbol = 0; symbol < symbolCount; ++symbol) {
            if (lengths[symbol] == width) {
                result.symbols[index++] = static_cast<std::uint16_t>(symbol);
            }
        }
    }
    return result;
}

template<std::size_t N, unsigned MaxBits>
[[nodiscard]] constexpr std::expected<unsigned, Error> decodeSymbol(BitReader& reader, const HuffmanDecoder<N, MaxBits>& decoder) noexcept {
    std::uint32_t code  = 0;
    std::uint32_t first = 0;
    std::size_t   index = 0;
    for (unsigned width = 1; width <= decoder.maxWidth; ++width) {
        const auto bit = reader.readBits(1);
        if (!bit) {
            return std::unexpected(bit.error());
        }
        code             = (code << 1U) | *bit;
        const auto count = decoder.counts[width];
        if (code >= first && code - first < count) {
            return decoder.symbols[index + code - first];
        }
        index += count;
        first = (first + count) << 1U;
    }
    return std::unexpected(Error::invalidCode);
}

[[nodiscard]] consteval auto makeFixedLiteralDecoder() {
    std::array<std::uint8_t, 288> lengths{};
    for (std::size_t i = 0; i <= 143UZ; ++i) {
        lengths[i] = 8;
    }
    for (std::size_t i = 144; i <= 255UZ; ++i) {
        lengths[i] = 9;
    }
    for (std::size_t i = 256; i <= 279UZ; ++i) {
        lengths[i] = 7;
    }
    for (std::size_t i = 280; i < lengths.size(); ++i) {
        lengths[i] = 8;
    }
    return *makeDecoder<288, 15>(lengths);
}

[[nodiscard]] consteval auto makeFixedDistanceDecoder() {
    std::array<std::uint8_t, 32> lengths{};
    lengths.fill(5);
    return *makeDecoder<32, 15>(lengths);
}

inline constexpr auto kFixedLiteralDecoder  = makeFixedLiteralDecoder();
inline constexpr auto kFixedDistanceDecoder = makeFixedDistanceDecoder();

struct CountingOutput {
    std::size_t count{};
    std::size_t historyBegin{};

    [[nodiscard]] constexpr std::size_t size() const noexcept { return count; }
    constexpr void                      resetHistory() noexcept { historyBegin = count; }

    [[nodiscard]] constexpr std::expected<void, Error> append(std::byte) noexcept {
        if (count == std::numeric_limits<std::size_t>::max()) {
            return std::unexpected(Error::integerOverflow);
        }
        ++count;
        return {};
    }

    [[nodiscard]] constexpr std::expected<void, Error> copy(std::size_t distance, std::size_t length) noexcept {
        if (distance == 0UZ || distance > count - historyBegin) {
            return std::unexpected(Error::invalidBackReference);
        }
        if (length > std::numeric_limits<std::size_t>::max() - count) {
            return std::unexpected(Error::integerOverflow);
        }
        count += length;
        return {};
    }
};

struct SpanOutput {
    std::span<std::byte> output;
    std::size_t          pos{};
    std::size_t          historyBegin{};

    [[nodiscard]] constexpr std::size_t size() const noexcept { return pos; }
    constexpr void                      resetHistory() noexcept { historyBegin = pos; }

    [[nodiscard]] constexpr std::expected<void, Error> append(std::byte value) noexcept {
        if (pos == output.size()) {
            return std::unexpected(Error::outputTooSmall);
        }
        output[pos++] = value;
        return {};
    }

    [[nodiscard]] constexpr std::expected<void, Error> copy(std::size_t distance, std::size_t length) noexcept {
        if (distance == 0UZ || distance > pos - historyBegin) {
            return std::unexpected(Error::invalidBackReference);
        }
        if (length > output.size() - pos) {
            return std::unexpected(Error::outputTooSmall);
        }
        for (std::size_t i = 0; i < length; ++i) {
            output[pos] = output[pos - distance];
            ++pos;
        }
        return {};
    }

    [[nodiscard]] constexpr std::span<const std::byte> range(std::size_t begin) const noexcept { return output.first(pos).subspan(begin); }
};

template<typename Output, std::size_t LiteralSymbols, std::size_t DistanceSymbols>
[[nodiscard]] constexpr std::expected<void, Error> decodeCompressedBlock(BitReader& reader, Output& output, const HuffmanDecoder<LiteralSymbols, 15>& literals, const HuffmanDecoder<DistanceSymbols, 15>& distances) noexcept {
    while (true) {
        const auto symbol = decodeSymbol(reader, literals);
        if (!symbol) {
            return std::unexpected(symbol.error());
        }
        if (*symbol < 256U) {
            if (const auto result = output.append(static_cast<std::byte>(*symbol)); !result) {
                return result;
            }
            continue;
        }
        if (*symbol == 256U) {
            return {};
        }
        if (*symbol > 285U) {
            return std::unexpected(Error::invalidLength);
        }

        const auto  lengthIndex = *symbol - 257U;
        std::size_t length      = kLengthBase[lengthIndex];
        if (const auto extra = kLengthExtra[lengthIndex]; extra != 0U) {
            const auto value = reader.readBits(extra);
            if (!value) {
                return std::unexpected(value.error());
            }
            length += *value;
        }

        const auto distanceSymbol = decodeSymbol(reader, distances);
        if (!distanceSymbol) {
            return std::unexpected(distanceSymbol.error());
        }
        if (*distanceSymbol >= kDistanceBase.size()) {
            return std::unexpected(Error::invalidDistance);
        }
        std::size_t distance = kDistanceBase[*distanceSymbol];
        if (const auto extra = kDistanceExtra[*distanceSymbol]; extra != 0U) {
            const auto value = reader.readBits(extra);
            if (!value) {
                return std::unexpected(value.error());
            }
            distance += *value;
        }
        if (const auto result = output.copy(distance, length); !result) {
            return result;
        }
    }
}

template<typename Output>
[[nodiscard]] constexpr std::expected<void, Error> decodeDynamicBlock(BitReader& reader, Output& output) noexcept {
    const auto literalBits  = reader.readBits(5);
    const auto distanceBits = reader.readBits(5);
    const auto codeBits     = reader.readBits(4);
    if (!literalBits || !distanceBits || !codeBits) {
        return std::unexpected(Error::truncatedInput);
    }
    const auto literalCount    = static_cast<std::size_t>(*literalBits + 257U);
    const auto distanceCount   = static_cast<std::size_t>(*distanceBits + 1U);
    const auto codeLengthCount = static_cast<std::size_t>(*codeBits + 4U);

    std::array<std::uint8_t, 19> codeLengthLengths{};
    for (std::size_t i = 0; i < codeLengthCount; ++i) {
        const auto length = reader.readBits(3);
        if (!length) {
            return std::unexpected(length.error());
        }
        codeLengthLengths[kCodeLengthOrder[i]] = static_cast<std::uint8_t>(*length);
    }
    const auto codeLengthDecoder = makeDecoder<19, 7>(codeLengthLengths);
    if (!codeLengthDecoder) {
        return std::unexpected(codeLengthDecoder.error());
    }

    std::array<std::uint8_t, 318> lengths{};
    const auto                    total = literalCount + distanceCount;
    std::size_t                   index = 0;
    while (index < total) {
        const auto symbol = decodeSymbol(reader, *codeLengthDecoder);
        if (!symbol) {
            return std::unexpected(symbol.error());
        }
        if (*symbol <= 15U) {
            lengths[index++] = static_cast<std::uint8_t>(*symbol);
            continue;
        }

        std::uint8_t value = 0;
        unsigned     repeatBase{};
        unsigned     repeatBits{};
        if (*symbol == 16U) {
            if (index == 0UZ) {
                return std::unexpected(Error::invalidHuffmanTree);
            }
            value      = lengths[index - 1UZ];
            repeatBase = 3;
            repeatBits = 2;
        } else if (*symbol == 17U) {
            repeatBase = 3;
            repeatBits = 3;
        } else if (*symbol == 18U) {
            repeatBase = 11;
            repeatBits = 7;
        } else {
            return std::unexpected(Error::invalidCode);
        }
        const auto extra = reader.readBits(repeatBits);
        if (!extra) {
            return std::unexpected(extra.error());
        }
        const auto repeat = static_cast<std::size_t>(repeatBase + *extra);
        if (repeat > total - index) {
            return std::unexpected(Error::invalidHuffmanTree);
        }
        for (std::size_t i = 0; i < repeat; ++i) {
            lengths[index++] = value;
        }
    }

    std::array<std::uint8_t, 288> literalLengths{};
    std::array<std::uint8_t, 32>  distanceLengths{};
    std::copy_n(lengths.begin(), literalCount, literalLengths.begin());
    std::copy_n(lengths.begin() + static_cast<std::ptrdiff_t>(literalCount), distanceCount, distanceLengths.begin());
    if (literalLengths[256] == 0U) {
        return std::unexpected(Error::invalidHuffmanTree);
    }
    const auto literalDecoder  = makeDecoder<288, 15>(literalLengths, literalCount);
    const auto distanceDecoder = makeDecoder<32, 15>(distanceLengths, distanceCount);
    if (!literalDecoder || !distanceDecoder) {
        return std::unexpected(Error::invalidHuffmanTree);
    }
    return decodeCompressedBlock(reader, output, *literalDecoder, *distanceDecoder);
}

template<typename Output>
[[nodiscard]] constexpr std::expected<std::size_t, Error> decodeDeflate(std::span<const std::byte> input, Output& output) noexcept {
    output.resetHistory();
    BitReader reader{.input = input};
    bool      final = false;
    while (!final) {
        const auto finalBit = reader.readBits(1);
        const auto block    = reader.readBits(2);
        if (!finalBit || !block) {
            return std::unexpected(Error::truncatedInput);
        }
        final = *finalBit != 0U;
        switch (*block) {
        case 0: {
            reader.alignByte();
            const auto lengthValue     = reader.readBits(16);
            const auto complementValue = reader.readBits(16);
            if (!lengthValue || !complementValue) {
                return std::unexpected(Error::truncatedInput);
            }
            const auto length = static_cast<std::uint16_t>(*lengthValue);
            if (static_cast<std::uint16_t>(~length) != static_cast<std::uint16_t>(*complementValue)) {
                return std::unexpected(Error::invalidStoredBlockLength);
            }
            for (std::size_t i = 0; i < length; ++i) {
                const auto value = reader.readBits(8);
                if (!value) {
                    return std::unexpected(value.error());
                }
                if (const auto result = output.append(static_cast<std::byte>(*value)); !result) {
                    return std::unexpected(result.error());
                }
            }
            break;
        }
        case 1:
            if (const auto result = decodeCompressedBlock(reader, output, kFixedLiteralDecoder, kFixedDistanceDecoder); !result) {
                return std::unexpected(result.error());
            }
            break;
        case 2:
            if (const auto result = decodeDynamicBlock(reader, output); !result) {
                return std::unexpected(result.error());
            }
            break;
        default: return std::unexpected(Error::invalidBlockType);
        }
    }
    reader.alignByte();
    return reader.pos;
}

[[nodiscard]] constexpr std::uint16_t readLittle16(const std::byte* p) noexcept { return static_cast<std::uint16_t>(std::to_integer<std::uint16_t>(p[0]) | (std::to_integer<std::uint16_t>(p[1]) << 8U)); }
[[nodiscard]] constexpr std::uint32_t readLittle32(const std::byte* p) noexcept { return std::to_integer<std::uint32_t>(p[0]) | (std::to_integer<std::uint32_t>(p[1]) << 8U) | (std::to_integer<std::uint32_t>(p[2]) << 16U) | (std::to_integer<std::uint32_t>(p[3]) << 24U); }
[[nodiscard]] constexpr std::uint32_t readBig32(const std::byte* p) noexcept { return (std::to_integer<std::uint32_t>(p[0]) << 24U) | (std::to_integer<std::uint32_t>(p[1]) << 16U) | (std::to_integer<std::uint32_t>(p[2]) << 8U) | std::to_integer<std::uint32_t>(p[3]); }

[[nodiscard]] constexpr std::uint32_t computeCrc32(std::span<const std::byte> input) noexcept {
    if consteval {
        return gr::crc::computeScalar<gr::crc::Flavour::CRC32_IEEE>(input);
    } else {
        return gr::crc::compute<gr::crc::Flavour::CRC32_IEEE>(input);
    }
}

template<typename Output>
[[nodiscard]] constexpr std::expected<std::size_t, Error> decodeRaw(std::span<const std::byte> input, Output& output) noexcept {
    const auto consumed = decodeDeflate(input, output);
    if (!consumed) {
        return std::unexpected(consumed.error());
    }
    if (*consumed != input.size()) {
        return std::unexpected(Error::trailingData);
    }
    return output.size();
}

template<typename Output>
[[nodiscard]] constexpr std::expected<std::size_t, Error> decodeZlib(std::span<const std::byte> input, Output& output) noexcept {
    if (input.size() < 2UZ) {
        return std::unexpected(Error::truncatedInput);
    }
    const auto cmf = std::to_integer<std::uint8_t>(input[0]);
    const auto flg = std::to_integer<std::uint8_t>(input[1]);
    if ((cmf & 0x0fU) != 8U || (cmf >> 4U) > 7U || ((static_cast<unsigned>(cmf) << 8U) + flg) % 31U != 0U) {
        return std::unexpected(Error::invalidHeader);
    }
    if ((flg & 0x20U) != 0U) {
        if (input.size() < 6UZ) {
            return std::unexpected(Error::truncatedInput);
        }
        return std::unexpected(Error::presetDictionaryRequired);
    }

    const auto outputBegin = output.size();
    const auto consumed    = decodeDeflate(input.subspan(2), output);
    if (!consumed) {
        return std::unexpected(consumed.error());
    }
    const auto trailer = 2UZ + *consumed;
    if (input.size() < trailer + 4UZ) {
        return std::unexpected(Error::truncatedInput);
    }
    if (input.size() != trailer + 4UZ) {
        return std::unexpected(Error::trailingData);
    }
    if constexpr (requires { output.range(outputBegin); }) {
        if (adler32(output.range(outputBegin)) != readBig32(input.data() + trailer)) {
            return std::unexpected(Error::checksumMismatch);
        }
    }
    return output.size();
}

template<typename Output>
[[nodiscard]] constexpr std::expected<std::size_t, Error> decodeGzip(std::span<const std::byte> input, Output& output) noexcept {
    std::size_t pos = 0;
    if (input.empty()) {
        return std::unexpected(Error::truncatedInput);
    }
    while (pos < input.size()) {
        const auto memberStart = pos;
        if (input.size() - pos < 2UZ) {
            return memberStart == 0UZ || input[pos] == std::byte{0x1f} ? std::unexpected(Error::truncatedInput) : std::unexpected(Error::trailingData);
        }
        if (input[pos] != std::byte{0x1f} || input[pos + 1UZ] != std::byte{0x8b}) {
            return memberStart == 0UZ ? std::unexpected(Error::invalidHeader) : std::unexpected(Error::trailingData);
        }
        if (input.size() - pos < 10UZ) {
            return std::unexpected(Error::truncatedInput);
        }
        if (input[pos + 2UZ] != std::byte{0x08}) {
            return std::unexpected(Error::unsupportedCompressionMethod);
        }
        const auto flags = std::to_integer<std::uint8_t>(input[pos + 3UZ]);
        if ((flags & 0xe0U) != 0U) {
            return std::unexpected(Error::reservedFlags);
        }
        pos += 10UZ;

        if ((flags & 0x04U) != 0U) {
            if (input.size() - pos < 2UZ) {
                return std::unexpected(Error::truncatedInput);
            }
            const auto extraLength = readLittle16(input.data() + pos);
            pos += 2UZ;
            if (input.size() - pos < extraLength) {
                return std::unexpected(Error::truncatedInput);
            }
            pos += extraLength;
        }
        for (const auto flag : {0x08U, 0x10U}) {
            if ((flags & flag) == 0U) {
                continue;
            }
            while (pos < input.size() && input[pos] != std::byte{0x00}) {
                ++pos;
            }
            if (pos == input.size()) {
                return std::unexpected(Error::truncatedInput);
            }
            ++pos;
        }
        if ((flags & 0x02U) != 0U) {
            if (input.size() - pos < 2UZ) {
                return std::unexpected(Error::truncatedInput);
            }
            const auto expected = readLittle16(input.data() + pos);
            const auto actual   = static_cast<std::uint16_t>(computeCrc32(input.subspan(memberStart, pos - memberStart)) & 0xffffU);
            if (expected != actual) {
                return std::unexpected(Error::checksumMismatch);
            }
            pos += 2UZ;
        }

        const auto outputBegin = output.size();
        const auto consumed    = decodeDeflate(input.subspan(pos), output);
        if (!consumed) {
            return std::unexpected(consumed.error());
        }
        pos += *consumed;
        if (input.size() - pos < 8UZ) {
            return std::unexpected(Error::truncatedInput);
        }
        if constexpr (requires { output.range(outputBegin); }) {
            if (computeCrc32(output.range(outputBegin)) != readLittle32(input.data() + pos)) {
                return std::unexpected(Error::checksumMismatch);
            }
        }
        if (static_cast<std::uint32_t>(output.size() - outputBegin) != readLittle32(input.data() + pos + 4UZ)) {
            return std::unexpected(Error::sizeMismatch);
        }
        pos += 8UZ;
    }
    return output.size();
}

template<typename Output>
[[nodiscard]] constexpr std::expected<std::size_t, Error> decompressInto(std::span<const std::byte> input, Output& output, Format format) noexcept {
    switch (format) {
    case Format::rawDeflate: return decodeRaw(input, output);
    case Format::zlib: return decodeZlib(input, output);
    case Format::gzip: return decodeGzip(input, output);
    }
    return std::unexpected(Error::invalidHeader);
}

} // namespace detail

template<auto Data, CompressionLevel Level = CompressionLevel::balanced>
[[nodiscard]] consteval auto gzip() {
    static_assert(Data.size() <= std::numeric_limits<std::uint32_t>::max(), "gzip ISIZE is limited to 32 bits for compile-time resources");
    constexpr auto tokens   = detail::tokenize<Data, Level>();
    constexpr auto coding   = detail::makeDynamicCoding(tokens);
    constexpr auto encoding = detail::selectEncoding<Data>(tokens, coding);
    constexpr auto size     = 18UZ + detail::deflateSize<Data>(encoding, tokens, coding);
    constexpr auto xfl      = Level == CompressionLevel::fast ? std::byte{0x04} : Level == CompressionLevel::best ? std::byte{0x02} : std::byte{0x00};

    std::array<std::byte, size> bytes{};
    detail::ArraySink<size>     sink{bytes};
    for (const auto byte : std::array{std::byte{0x1f}, std::byte{0x8b}, std::byte{0x08}, std::byte{0x00}, std::byte{0x00}, std::byte{0x00}, std::byte{0x00}, std::byte{0x00}, xfl, std::byte{0xff}}) {
        sink.put(byte);
    }
    detail::emitDeflate<Data>(sink, encoding, tokens, coding);
    detail::putLittle32(sink, gr::crc::computeScalar<gr::crc::Flavour::CRC32_IEEE>(Data));
    detail::putLittle32(sink, static_cast<std::uint32_t>(Data.size()));
    return bytes;
}

[[nodiscard]] constexpr std::expected<std::size_t, Error> decompress(std::span<const std::byte> input, std::span<std::byte> output, Format format) noexcept {
    detail::SpanOutput writer{.output = output};
    return detail::decompressInto(input, writer, format);
}

[[nodiscard]] inline std::expected<std::vector<std::byte>, Error> decompress(std::span<const std::byte> input, Format format) noexcept {
    detail::CountingOutput counter{};
    const auto             counted = detail::decompressInto(input, counter, format);
    if (!counted) {
        return std::unexpected(counted.error());
    }
    try {
        std::vector<std::byte> output(*counted);
        const auto             written = decompress(input, output, format);
        if (!written) {
            return std::unexpected(written.error());
        }
        if (*written != output.size()) {
            return std::unexpected(Error::sizeMismatch);
        }
        return output;
    } catch (const std::bad_alloc&) {
        return std::unexpected(Error::allocationFailed);
    } catch (const std::length_error&) {
        return std::unexpected(Error::allocationFailed);
    }
}

template<auto Packed, std::size_t OriginalSize>
struct CompressedBlob {
    [[nodiscard]] static constexpr std::size_t compressedSize() noexcept { return Packed.size(); }
    [[nodiscard]] static constexpr std::size_t uncompressedSize() noexcept { return OriginalSize; }
    [[nodiscard]] static constexpr const auto& compressed() noexcept { return Packed; }

    [[nodiscard]] static const std::vector<std::byte>& decompress() {
        static const std::vector<std::byte> cache = [] {
            std::vector<std::byte> output(OriginalSize);
            const auto             result = gr::compression::decompress(Packed, output, Format::gzip);
            assert(result.has_value() && *result == OriginalSize);
            return result && *result == OriginalSize ? std::move(output) : std::vector<std::byte>{};
        }();
        return cache;
    }

    [[nodiscard]] std::span<const std::byte> bytes() const { return decompress(); }
};

template<auto Packed, std::size_t OriginalSize>
struct CompressedText {
    [[nodiscard]] static constexpr std::size_t compressedSize() noexcept { return Packed.size(); }
    [[nodiscard]] static constexpr std::size_t uncompressedSize() noexcept { return OriginalSize; }
    [[nodiscard]] static constexpr const auto& compressed() noexcept { return Packed; }

    [[nodiscard]] static const std::string& str() {
        static const std::string cache = [] {
            std::string output(OriginalSize, '\0');
            const auto  result = gr::compression::decompress(Packed, std::as_writable_bytes(std::span(output)), Format::gzip);
            assert(result.has_value() && *result == OriginalSize);
            return result && *result == OriginalSize ? std::move(output) : std::string{};
        }();
        return cache;
    }

    [[nodiscard]] std::string_view view() const { return str(); }
    [[nodiscard]] const char*      c_str() const { return str().c_str(); }

    [[nodiscard]] operator std::string_view() const { return view(); }
};

template<auto Data, CompressionLevel Level = CompressionLevel::balanced>
[[nodiscard]] consteval auto makeCompressedBlob() {
    constexpr auto packed = gzip<Data, Level>();
    return CompressedBlob<packed, Data.size()>{};
}

template<gr::meta::fixed_string S, CompressionLevel Level = CompressionLevel::balanced>
[[nodiscard]] consteval auto makeCompressedText() {
    constexpr auto packed = gzip<detail::staticBytesFromString<S>(), Level>();
    return CompressedText<packed, S.size()>{};
}

namespace literals {
template<gr::meta::fixed_string S>
[[nodiscard]] consteval auto operator""_gzip() {
    return makeCompressedText<S>();
}
} // namespace literals

} // namespace gr::compression

#endif // GNURADIO_COMPRESSION_HPP
