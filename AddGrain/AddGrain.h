#pragma once

#include <type_traits>
#include <vector>

#include <VapourSynth4.h>
#include <VSHelper4.h>

#ifdef ADDGRAIN_X86
#include "VCL2/vectorclass.h"
#endif

// max # of noise planes
static constexpr int MAXP = 2;

// offset in pixels of the fake plane MAXP relative to plane MAXP-1
static constexpr int OFFSET_FAKEPLANE = 32;

struct AddGrainData final {
    VSNode* node;
    const VSVideoInfo* vi;
    float var, uvar, hcorr, vcorr;
    bool constant;
    bool process[3];
    int storedFrames, step, peak;
    std::vector<uint8_t> pNoiseSeeds;
    long idum;
    int nStride[MAXP], nHeight[MAXP], nSize[MAXP];
    void* pN[MAXP];
    void (*updateFrame)(const void* _srcp, void* _dstp, const int width, const int height, const ptrdiff_t stride, const int noisePlane, const int noiseOffs, const AddGrainData* const VS_RESTRICT d) noexcept;
};
