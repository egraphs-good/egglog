#include <algorithm>
#include <cfenv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include <mpfr.h>

using Kernel = uint16_t (*)(uint16_t);
struct Entry { const char *function; const char *schedule; Kernel run; };
#include "registry.inc"

constexpr uint32_t code_count = 1u << 16;

// MPFR evaluates the elementary function directly at precision 8 (RNE).
// Apart from exact log(1) = +0, every positive finite input yields a normal,
// finite bfloat16 logarithm. Its exact conversion to float is therefore safe.
class Oracle {
    mpfr_t input, answer;
public:
    Oracle() { mpfr_init2(input, 8); mpfr_init2(answer, 8); }
    ~Oracle() { mpfr_clear(input); mpfr_clear(answer); }
    Oracle(const Oracle&) = delete;
    uint16_t eval(bool base2, uint16_t bits) {
        if ((bits & 0x7fff) == 0) return 0xff80;
        if (bits == 0x7f80) return bits;
        if (bits > 0x7f80) return 0xffff;
        unsigned exponent = bits >> 7, fraction = bits & 127;
        unsigned significand = exponent ? 128 + fraction : fraction;
        int shift = exponent ? static_cast<int>(exponent) - 134 : -133;
        mpfr_set_ui_2exp(input, significand, shift, MPFR_RNDN);
        if (base2) mpfr_log2(answer, input, MPFR_RNDN);
        else mpfr_log10(answer, input, MPFR_RNDN);
        float exact = mpfr_get_flt(answer, MPFR_RNDN);
        static_assert(std::numeric_limits<float>::is_iec559 && sizeof(float) == sizeof(uint32_t));
        uint32_t word;
        std::memcpy(&word, &exact, sizeof(word));
        if ((word & 0xffff) || (mpfr_zero_p(answer) == 0 && !std::isnormal(exact)))
            throw std::runtime_error("MPFR oracle result outside the exact conversion range");
        return static_cast<uint16_t>(word >> 16);
    }
};

static uint64_t fingerprint(uint64_t hash, uint16_t value) {
    return (hash ^ value) * UINT64_C(1099511628211);
}

static void write_words(const std::string& path, const std::vector<uint16_t>& words) {
    std::ofstream out(path, std::ios::binary);
    for (uint16_t word : words) {
        out.put(static_cast<char>(word & 0xff));
        out.put(static_cast<char>(word >> 8));
    }
    out.close();
    if (!out) throw std::runtime_error("could not write " + path);
}

static int validate(const std::string& directory) {
    Oracle oracle;
    int failed = 0;
    std::puts("function,schedule,inputs,positive_normal,positive_subnormal,other_encodings,oracle_hash,result_hash,mismatches");
    for (const char *function : {"log2", "log10"}) {
        std::vector<uint16_t> expected(code_count);
        uint64_t oracle_hash = UINT64_C(14695981039346656037);
        for (uint32_t i = 0; i < code_count; ++i) {
            expected[i] = oracle.eval(std::strcmp(function, "log2") == 0, static_cast<uint16_t>(i));
            oracle_hash = fingerprint(oracle_hash, expected[i]);
        }
        write_words(directory + "/oracle_" + function + ".bin", expected);
        for (const auto& entry : entries) {
            if (std::strcmp(entry.function, function) != 0) continue;
            uint64_t hash = UINT64_C(14695981039346656037);
            uint32_t different = 0;
            for (uint32_t i = 0; i < code_count; ++i) {
                uint16_t got = entry.run(static_cast<uint16_t>(i));
                hash = fingerprint(hash, got);
                if (got != expected[i] && ++different <= 5)
                    std::fprintf(stderr, "%s/%s input=%04x expected=%04x got=%04x\n",
                                 function, entry.schedule, i, expected[i], got);
            }
            std::printf("%s,%s,%u,32512,127,32897,%016llx,%016llx,%u\n", function, entry.schedule,
                        code_count, (unsigned long long)oracle_hash, (unsigned long long)hash, different);
            failed |= different != 0;
        }
    }
    return failed;
}

static const Entry* find_entry(const char *function, const char *schedule) {
    for (const auto& e : entries)
        if (!std::strcmp(e.function, function) && !std::strcmp(e.schedule, schedule)) return &e;
    return nullptr;
}

enum class Mode { finite, all, chain };

// Fisher–Yates with an explicitly specified PRNG keeps the same input order on
// different standard libraries. Positive finite inputs include all subnormals.
static std::vector<uint16_t> inputs(Mode mode) {
    size_t size = mode == Mode::all ? code_count : 0x7f7fu;
    std::vector<uint16_t> values(size);
    for (size_t i = 0; i < size; ++i) values[i] = static_cast<uint16_t>(mode == Mode::all ? i : i+1);
    uint32_t state = 0xd3560a75u;
    for (size_t i = size; i > 1; --i) {
        state ^= state << 13; state ^= state >> 17; state ^= state << 5;
        std::swap(values[i-1], values[state % i]);
    }
    if (mode == Mode::chain) {
        // Power-of-two address mask avoids a division in the dependency chain.
        values.reserve(code_count);
        for (size_t i = size; i < code_count; ++i) values.push_back(values[i % size]);
    }
    return values;
}

static uint64_t evaluate(Kernel fn, const std::vector<uint16_t>& values, Mode mode, uint32_t sweeps) {
    uint64_t sum = 0;
    if (mode == Mode::chain) {
        uint32_t result = 0, step = 0;
        for (uint32_t r = 0; r < sweeps; ++r)
            for (size_t i = 0; i < values.size(); ++i) {
                step += 40503;
                result = fn(values[(result + step) & (code_count - 1)]);
                sum += result;
            }
    } else {
        for (uint32_t r = 0; r < sweeps; ++r)
            for (uint16_t bits : values) sum += fn(bits);
    }
    return sum;
}

int main(int argc, char **argv) {
    if (std::fegetround() != FE_TONEAREST) throw std::runtime_error("expected round-to-nearest");
    if (argc == 3 && !std::strcmp(argv[1], "validate")) {
        write_words(std::string(argv[2]) + "/inputs_all.bin", inputs(Mode::all));
        write_words(std::string(argv[2]) + "/inputs_positive.bin", inputs(Mode::finite));
        write_words(std::string(argv[2]) + "/inputs_chain.bin", inputs(Mode::chain));
        return validate(argv[2]);
    }
    if (argc != 5) return 2;
    const Entry *entry = find_entry(argv[1], argv[2]);
    if (!entry) return 2;
    Mode mode;
    if (!std::strcmp(argv[3], "positive")) mode = Mode::finite;
    else if (!std::strcmp(argv[3], "all")) mode = Mode::all;
    else if (!std::strcmp(argv[3], "chain")) mode = Mode::chain;
    else return 2;
    uint32_t sweeps = static_cast<uint32_t>(std::stoul(argv[4]));
    auto values = inputs(mode);
    volatile uint64_t warm = evaluate(entry->run, values, mode, 8);
    (void)warm;
    auto start = std::chrono::steady_clock::now();
    uint64_t sum = evaluate(entry->run, values, mode, sweeps);
    auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-start);
    std::printf("%llu,%llu,%llu\n", (unsigned long long)elapsed.count(),
                (unsigned long long)values.size() * sweeps, (unsigned long long)sum);
}
