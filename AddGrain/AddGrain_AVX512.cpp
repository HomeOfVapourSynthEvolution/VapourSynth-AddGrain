#ifdef ADDGRAIN_X86
#include "AddGrain.h"

template<typename pixel_t, typename noise_t>
void updateFrame_avx512(const void * _srcp, void * _dstp, const int width, const int height, const int stride, const int noisePlane, const int noiseOffs,
                        const AddGrainData * const VS_RESTRICT d) noexcept {
    const pixel_t * srcp = reinterpret_cast<const pixel_t *>(_srcp);
    pixel_t * dstp = reinterpret_cast<pixel_t *>(_dstp);
    const noise_t * pNW = reinterpret_cast<noise_t *>(d->pN[noisePlane]) + noiseOffs;

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x += d->step) {
            if constexpr (std::is_same_v<pixel_t, uint8_t>) {
                const Vec64c sign = 0x80;
                Vec64c src = Vec64c().load_a(srcp + x);
                const Vec64c noise = Vec64c().load(pNW + x);
                src ^= sign;
                src = add_saturated(src, noise);
                src ^= sign;
                src.store_nt(dstp + x);
            } else if constexpr (std::is_same_v<pixel_t, uint16_t>) {
                const Vec32s sign = 0x8000;
                Vec32s src = Vec32s().load_a(srcp + x);
                const Vec32s noise = Vec32s().load(pNW + x);
                src ^= sign;
                src = add_saturated(src, noise);
                src ^= sign;
                min(Vec32us(src), d->peak).store_nt(dstp + x);
            } else {
                Vec16f src = Vec16f().load_a(srcp + x);
                const Vec16f noise = Vec16f().load(pNW + x);
                (src + noise).store_nt(dstp + x);
            }
        }

        srcp += stride;
        dstp += stride;
        pNW += d->nStride[noisePlane];
    }
}

template void updateFrame_avx512<uint8_t, int8_t>(const void * _srcp, void * _dstp, const int width, const int height, const int stride, const int noisePlane, const int noiseOffs, const AddGrainData * const VS_RESTRICT d) noexcept;
template void updateFrame_avx512<uint16_t, int16_t>(const void * _srcp, void * _dstp, const int width, const int height, const int stride, const int noisePlane, const int noiseOffs, const AddGrainData * const VS_RESTRICT d) noexcept;
template void updateFrame_avx512<float, float>(const void * _srcp, void * _dstp, const int width, const int height, const int stride, const int noisePlane, const int noiseOffs, const AddGrainData * const VS_RESTRICT d) noexcept;
#endif
