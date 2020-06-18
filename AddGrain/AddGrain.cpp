// Copyright (c) 2002 Tom Barry.  All rights reserved.
//      trbarry@trbarry.com
//  modified by Foxyshadis
//      foxyshadis@hotmail.com
//  modified by Firesledge
//      http://ldesoras.free.fr
//  modified by LaTo INV.
//      http://forum.doom9.org/member.php?u=131032
//  VapourSynth port by HolyWu
//      https://github.com/HomeOfVapourSynthEvolution/VapourSynth-AddGrain
//
//  This program is free software: you can redistribute it and/or modify
//  it under the terms of the GNU General Public License as published by
//  the Free Software Foundation, either version 3 of the License, or
//  (at your option) any later version.
//
//  This program is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
//
//  You should have received a copy of the GNU General Public License
//  along with this program.  If not, see <https://www.gnu.org/licenses/>.

#include <cmath>
#include <ctime>

#include <algorithm>
#include <memory>
#include <string>

#include "AddGrain.h"

#ifdef ADDGRAIN_X86
template<typename pixel_t, typename noise_t> extern void updateFrame_sse2(const void * _srcp, void * _dstp, const int width, const int height, const int stride, const int noisePlane, const int noiseOffs, const AddGrainData * const VS_RESTRICT d) noexcept;
template<typename pixel_t, typename noise_t> extern void updateFrame_avx2(const void * _srcp, void * _dstp, const int width, const int height, const int stride, const int noisePlane, const int noiseOffs, const AddGrainData * const VS_RESTRICT d) noexcept;
template<typename pixel_t, typename noise_t> extern void updateFrame_avx512(const void * _srcp, void * _dstp, const int width, const int height, const int stride, const int noisePlane, const int noiseOffs, const AddGrainData * const VS_RESTRICT d) noexcept;
#endif

template<typename T>
static T getArg(const VSAPI * vsapi, const VSMap * map, const char * key, const T defaultValue) noexcept {
    T arg{};
    int err{};

    if constexpr (std::is_same_v<T, bool>)
        arg = !!vsapi->propGetInt(map, key, 0, &err);
    else if constexpr (std::is_same_v<T, int>)
        arg = int64ToIntS(vsapi->propGetInt(map, key, 0, &err));
    else if constexpr (std::is_same_v<T, int64_t>)
        arg = vsapi->propGetInt(map, key, 0, &err);
    else if constexpr (std::is_same_v<T, float>)
        arg = static_cast<float>(vsapi->propGetFloat(map, key, 0, &err));
    else if constexpr (std::is_same_v<T, double>)
        arg = vsapi->propGetFloat(map, key, 0, &err);

    if (err)
        arg = defaultValue;

    return arg;
}

static inline long fastUniformRandL(long & idum) noexcept {
    return idum = 1664525L * idum + 1013904223L;
}

// very fast & reasonably random
static inline float fastUniformRandF(long & idum) noexcept {
    // work with 32-bit IEEE floating point only!
    fastUniformRandL(idum);
    const unsigned long itemp = 0x3f800000 | (0x007fffff & idum);
    return *reinterpret_cast<const float *>(&itemp) - 1.0f;
}

static inline float gaussianRand(bool & iset, float & gset, long & idum) noexcept {
    float fac, rsq, v1, v2;

    // return saved second
    if (iset) {
        iset = false;
        return gset;
    }

    do {
        v1 = 2.0f * fastUniformRandF(idum) - 1.0f;
        v2 = 2.0f * fastUniformRandF(idum) - 1.0f;
        rsq = v1 * v1 + v2 * v2;
    } while (rsq >= 1.0f || rsq == 0.0f);

    fac = std::sqrt(-2.0f * std::log(rsq) / rsq);

    // function generates two values every iteration, so save one for later
    gset = v1 * fac;
    iset = true;

    return v2 * fac;
}

static inline float gaussianRand(const float mean, const float variance, bool & iset, float & gset, long & idum) noexcept {
    return (variance == 0.0f) ? mean : gaussianRand(iset, gset, idum) * std::sqrt(variance) + mean;
}

template<typename noise_t>
static void generateNoise(const int planesNoise, const float scale, AddGrainData * const VS_RESTRICT d) noexcept {
    float nRep[MAXP];
    for (int i = 0; i < MAXP; i++)
        nRep[i] = d->constant ? 1.0f : 2.0f;

    std::vector<float> lastLine(d->nStride[0]); // assume plane 0 is the widest one
    constexpr float mean = 0.0f;
    const float pvar[] = { d->var, d->uvar };
    bool iset = false;
    float gset;
    auto pns = d->pNoiseSeeds.begin();

    for (int plane = 0; plane < planesNoise; plane++) {
        int h = static_cast<int>(std::ceil(d->nHeight[plane] * nRep[plane]));
        if (planesNoise == 2 && plane == 1) {
            // fake plane needs at least one more row, and more if the rows are too small. round to the upper number
            h += (OFFSET_FAKEPLANE + d->nStride[plane] - 1) / d->nStride[plane];
        }
        d->nSize[plane] = d->nStride[plane] * h;

        // allocate space for noise
        d->pN[plane] = malloc(d->nSize[plane] * sizeof(noise_t));

        for (int x = 0; x < d->nStride[plane]; x++)
            lastLine[x] = gaussianRand(mean, pvar[plane], iset, gset, d->idum); // things to vertically smooth against

        for (int y = 0; y < h; y++) {
            noise_t * pNW = reinterpret_cast<noise_t *>(d->pN[plane]) + d->nStride[plane] * y;
            float lastr = gaussianRand(mean, pvar[plane], iset, gset, d->idum); // something to horiz smooth against

            for (int x = 0; x < d->nStride[plane]; x++) {
                float r = gaussianRand(mean, pvar[plane], iset, gset, d->idum);

                r = lastr * d->hcorr + r * (1.0f - d->hcorr); // horizontal correlation
                lastr = r;

                r = lastLine[x] * d->vcorr + r * (1.0f - d->vcorr); // vert corr
                lastLine[x] = r;

                // set noise block
                if constexpr (std::is_integral_v<noise_t>)
                    *pNW++ = static_cast<noise_t>(std::round(r * scale));
                else
                    *pNW++ = r * scale;
            }
        }

        for (int x = d->storedFrames; x > 0; x--)
            *pns++ = fastUniformRandL(d->idum) & 0xff; // insert seed, to keep cache happy
    }
}

// on input, plane is the frame plane index (if applicable, 0 otherwise), and on output, it contains the selected noise plane
static void setRand(int & plane, int & noiseOffs, const int frameNumber, AddGrainData * const VS_RESTRICT d) noexcept {
    if (d->constant) {
        // force noise to be identical every frame
        if (plane >= MAXP) {
            plane = MAXP - 1;
            noiseOffs = OFFSET_FAKEPLANE;
        }
    } else {
        // pull seed back out, to keep cache happy
        const int seedIndex = frameNumber % d->storedFrames;
        const int p0 = d->pNoiseSeeds[seedIndex];

        if (plane == 0) {
            d->idum = p0;
        } else {
            d->idum = d->pNoiseSeeds[seedIndex + d->storedFrames];
            if (plane == 2) {
                // the trick to needing only 2 planes ^.~
                d->idum ^= p0;
                plane--;
            }
        }

        // start noise at random qword in top half of noise area
        noiseOffs = static_cast<int>(fastUniformRandF(d->idum) * d->nSize[plane] / MAXP) & 0xfffffff8;
    }
}

template<typename pixel_t, typename noise_t>
static void updateFrame_c(const void * _srcp, void * _dstp, const int width, const int height, const int stride, const int noisePlane, const int noiseOffs,
                          const AddGrainData * const VS_RESTRICT d) noexcept {
    const pixel_t * srcp = reinterpret_cast<const pixel_t *>(_srcp);
    pixel_t * VS_RESTRICT dstp = reinterpret_cast<pixel_t *>(_dstp);
    const noise_t * pNW = reinterpret_cast<noise_t *>(d->pN[noisePlane]) + noiseOffs;

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            if constexpr (std::is_integral_v<pixel_t>)
                dstp[x] = std::clamp(srcp[x] + pNW[x], 0, d->peak);
            else
                dstp[x] = srcp[x] + pNW[x];
        }

        srcp += stride;
        dstp += stride;
        pNW += d->nStride[noisePlane];
    }
}

static void VS_CC addgrainInit(VSMap * in, VSMap * out, void ** instanceData, VSNode * node, VSCore * core, const VSAPI * vsapi) {
    AddGrainData * d = static_cast<AddGrainData *>(*instanceData);
    vsapi->setVideoInfo(d->vi, 1, node);
}

static const VSFrameRef * VS_CC addgrainGetFrame(int n, int activationReason, void ** instanceData, void ** frameData, VSFrameContext * frameCtx, VSCore * core, const VSAPI * vsapi) {
    AddGrainData * d = static_cast<AddGrainData *>(*instanceData);

    if (activationReason == arInitial) {
        vsapi->requestFrameFilter(n, d->node, frameCtx);
    } else if (activationReason == arAllFramesReady) {
        const VSFrameRef * src = vsapi->getFrameFilter(n, d->node, frameCtx);
        const VSFrameRef * fr[] = { d->process[0] ? nullptr : src, d->process[1] ? nullptr : src, d->process[2] ? nullptr : src };
        const int pl[] = { 0, 1, 2 };
        VSFrameRef * dst = vsapi->newVideoFrame2(d->vi->format, d->vi->width, d->vi->height, fr, pl, src, core);

        for (int plane = 0; plane < d->vi->format->numPlanes; plane++) {
            if (d->process[plane]) {
                const int width = vsapi->getFrameWidth(src, plane);
                const int height = vsapi->getFrameHeight(src, plane);
                const int stride = vsapi->getStride(src, plane) / d->vi->format->bytesPerSample;
                const uint8_t * srcp = vsapi->getReadPtr(src, plane);
                uint8_t * dstp = vsapi->getWritePtr(dst, plane);

                int noisePlane = (d->vi->format->colorFamily == cmRGB) ? 0 : plane;
                int noiseOffs = 0;
                setRand(noisePlane, noiseOffs, n, d); // seed randomness w/ plane & frame
                d->updateFrame(srcp, dstp, width, height, stride, noisePlane, noiseOffs, d);
            }
        }

        vsapi->freeFrame(src);
        return dst;
    }

    return nullptr;
}

static void VS_CC addgrainFree(void * instanceData, VSCore * core, const VSAPI * vsapi) {
    AddGrainData * d = static_cast<AddGrainData *>(instanceData);

    vsapi->freeNode(d->node);

    for (int i = 0; i < MAXP; i++)
        free(d->pN[i]);

    delete d;
}

static void VS_CC addgrainCreate(const VSMap * in, VSMap * out, void * userData, VSCore * core, const VSAPI * vsapi) {
    using namespace std::literals;

    std::unique_ptr<AddGrainData> d = std::make_unique<AddGrainData>();

    try {
        d->node = vsapi->propGetNode(in, "clip", 0, nullptr);
        d->vi = vsapi->getVideoInfo(d->node);

        if (!isConstantFormat(d->vi) ||
            (d->vi->format->sampleType == stInteger && d->vi->format->bitsPerSample > 16) ||
            (d->vi->format->sampleType == stFloat && d->vi->format->bitsPerSample != 32))
            throw "only constant format 8-16 bit integer and 32 bit float input supported"sv;

        d->var = getArg(vsapi, in, "var", 1.0f);
        d->uvar = getArg(vsapi, in, "uvar", 0.0f);
        d->hcorr = getArg(vsapi, in, "hcorr", 0.0f);
        d->vcorr = getArg(vsapi, in, "vcorr", 0.0f);
        long seed = getArg(vsapi, in, "seed", -1);
        d->constant = getArg(vsapi, in, "constant", false);
        const int opt = getArg(vsapi, in, "opt", 0);

        if (d->hcorr < 0.0f || d->hcorr > 1.0f || d->vcorr < 0.0f || d->vcorr > 1.0f)
            throw "hcorr and vcorr must be between 0.0 and 1.0 (inclusive)"sv;

        if (opt < 0 || opt > 4)
            throw "opt must be 0, 1, 2, 3, or 4"sv;

        {
            if (d->vi->format->bytesPerSample == 1)
                d->updateFrame = updateFrame_c<uint8_t, int8_t>;
            else if (d->vi->format->bytesPerSample == 2)
                d->updateFrame = updateFrame_c<uint16_t, int16_t>;
            else
                d->updateFrame = updateFrame_c<float, float>;

#ifdef ADDGRAIN_X86
            const int iset = instrset_detect();
            if ((opt == 0 && iset >= 10) || opt == 4) {
                if (d->vi->format->bytesPerSample == 1) {
                    d->updateFrame = updateFrame_avx512<uint8_t, int8_t>;
                    d->step = 64;
                } else if (d->vi->format->bytesPerSample == 2) {
                    d->updateFrame = updateFrame_avx512<uint16_t, int16_t>;
                    d->step = 32;
                } else {
                    d->updateFrame = updateFrame_avx512<float, float>;
                    d->step = 16;
                }
            } else if ((opt == 0 && iset >= 8) || opt == 3) {
                if (d->vi->format->bytesPerSample == 1) {
                    d->updateFrame = updateFrame_avx2<uint8_t, int8_t>;
                    d->step = 32;
                } else if (d->vi->format->bytesPerSample == 2) {
                    d->updateFrame = updateFrame_avx2<uint16_t, int16_t>;
                    d->step = 16;
                } else {
                    d->updateFrame = updateFrame_avx2<float, float>;
                    d->step = 8;
                }
            } else if ((opt == 0 && iset >= 2) || opt == 2) {
                if (d->vi->format->bytesPerSample == 1) {
                    d->updateFrame = updateFrame_sse2<uint8_t, int8_t>;
                    d->step = 16;
                } else if (d->vi->format->bytesPerSample == 2) {
                    d->updateFrame = updateFrame_sse2<uint16_t, int16_t>;
                    d->step = 8;
                } else {
                    d->updateFrame = updateFrame_sse2<float, float>;
                    d->step = 4;
                }
            }
#endif
        }

        float scale;
        if (d->vi->format->sampleType == stInteger) {
            d->peak = (1 << d->vi->format->bitsPerSample) - 1;
            scale = d->peak / 255.0f;
        } else {
            scale = 1.0f / 255.0f;
        }

        int planesNoise = 1;
        d->nStride[0] = (d->vi->width + 63) & ~63; // first plane
        d->nHeight[0] = d->vi->height;
        if (d->vi->format->colorFamily == cmGray) {
            d->uvar = 0.0f;
        } else if (d->vi->format->colorFamily == cmRGB) {
            d->uvar = d->var;
        } else {
            planesNoise = 2;
            d->nStride[1] = ((d->vi->width >> d->vi->format->subSamplingW) + 63) & ~63; // second and third plane
            d->nHeight[1] = d->vi->height >> d->vi->format->subSamplingH;
        }

        if (d->var <= 0.0f && d->uvar <= 0.0f) {
            vsapi->propSetNode(out, "clip", d->node, paReplace);
            vsapi->freeNode(d->node);
            return;
        }

        d->process[0] = d->var > 0.0f;
        d->process[1] = d->process[2] = d->uvar > 0.0f;

        if (seed < 0)
            seed = static_cast<long>(std::time(nullptr)); // init random
        d->idum = seed;

        d->storedFrames = std::min(d->vi->numFrames, 256);
        d->pNoiseSeeds.resize(d->storedFrames * planesNoise);

        if (d->vi->format->bytesPerSample == 1)
            generateNoise<int8_t>(planesNoise, scale, d.get());
        else if (d->vi->format->bytesPerSample == 2)
            generateNoise<int16_t>(planesNoise, scale, d.get());
        else
            generateNoise<float>(planesNoise, scale, d.get());
    } catch (const std::string_view & error) {
        vsapi->setError(out, ("AddGrain: "s + error.data()).c_str());
        vsapi->freeNode(d->node);
        return;
    }

    vsapi->createFilter(in, out, "AddGrain", addgrainInit, addgrainGetFrame, addgrainFree, fmParallel, 0, d.release(), core);
}

//////////////////////////////////////////
// Init

VS_EXTERNAL_API(void) VapourSynthPluginInit(VSConfigPlugin configFunc, VSRegisterFunction registerFunc, VSPlugin * plugin) {
    configFunc("com.holywu.addgrain", "grain", "Add some correlated color gaussian noise", VAPOURSYNTH_API_VERSION, 1, plugin);
    registerFunc("Add",
                 "clip:clip;"
                 "var:float:opt;"
                 "uvar:float:opt;"
                 "hcorr:float:opt;"
                 "vcorr:float:opt;"
                 "seed:int:opt;"
                 "constant:int:opt;"
                 "opt:int:opt;",
                 addgrainCreate, nullptr, plugin);
}
