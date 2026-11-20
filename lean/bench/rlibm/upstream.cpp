// Compile this adapter separately for each unchanged source. In both cases the
// compiler sees the bfloat16 routine and ABI adapter in the same translation unit.
#if RLIBM_BASE == 2
#include "source/bfloat16/log2.cpp"
extern "C" uint16_t upstream_log2(uint16_t bits) { return rlibm_log2(bfloat16(bits)).val; }
#elif RLIBM_BASE == 10
#include "source/bfloat16/log10.cpp"
extern "C" uint16_t upstream_log10(uint16_t bits) { return rlibm_log10(bfloat16(bits)).val; }
#else
#error "unsupported logarithm base"
#endif
