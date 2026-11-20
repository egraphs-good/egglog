#ifndef VERIFIED_RLIBM_KERNEL_H
#define VERIFIED_RLIBM_KERNEL_H
#include <stdint.h>
#include "rlibm_coefficients.h"

typedef uint64_t (*rlibm_accumulator)(uint64_t, uint64_t, uint64_t, uint64_t,
                                      uint64_t, uint64_t, uint64_t);

/* Unsigned modular arithmetic implements every extracted accumulator. The
   Lean bound on the final signed magnitude makes its sign bit unambiguous. */
static inline uint16_t rlibm_round_q40(uint64_t z) {
    if (z == 0) return 0;
    unsigned negative = (unsigned)(z >> 63);
    uint64_t n = negative ? 0 - z : z;
    unsigned s = 63u - (unsigned)__builtin_clzll(n);
    uint64_t significand;
    if (s < 7) {
        significand = n << (7u - s);
    } else {
        unsigned cut = s - 7u;
        uint64_t unit = UINT64_C(1) << cut;
        uint64_t high = n >> cut;
        uint64_t twice_rem = 2 * (n & (unit - 1));
        significand = high + (twice_rem > unit || (twice_rem == unit && (high & 1)));
    }
    unsigned bits = (s + 86u) * 128u + (unsigned)significand;
    if (bits > 0x7f80u) bits = 0x7f80u;
    return (uint16_t)((negative << 15) + bits);
}

static inline uint16_t rlibm_execute(uint16_t raw, unsigned base, rlibm_accumulator fn,
                                     unsigned use_quadratic) {
    if ((raw & 0x7fffu) == 0) return 0xff80u;
    if (raw == 0x7f80u) return raw;
    if (raw > 0x7f80u) return 0xffffu;
    unsigned field = raw >> 7;
    unsigned m = raw & 127u;
    int e;
    if (field == 0) {
        unsigned shift = (unsigned)__builtin_clz(m) - (sizeof(unsigned) * 8u - 8u);
        e = -126 - (int)shift;
        m = (m << shift) - 128u;
    } else {
        e = (int)field - 127;
    }
    uint64_t k = base == 2 ? rlibm_step2 : rlibm_step10;
    if (use_quadratic) {
        const uint64_t *c = (base == 2 ? rlibm_quad2 : rlibm_quad10)[m / 8];
        uint64_t t = m % 8;
        return rlibm_round_q40((uint64_t)e * k + (c[0] + t * (c[1] + t * c[2])));
    }
    const uint64_t *c = (base == 2 ? rlibm_coef2 : rlibm_coef10)[m / 16];
    return rlibm_round_q40(fn((uint64_t)e, k, m % 16, c[0], c[1], c[2], c[3]));
}

static inline uint16_t rlibm_run(uint16_t raw, unsigned base, rlibm_accumulator fn) {
    return rlibm_execute(raw, base, fn, 0);
}

static inline uint16_t rlibm_run_quadratic(uint16_t raw, unsigned base) {
    return rlibm_execute(raw, base, (rlibm_accumulator)0, 1);
}
#endif
