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

using namespace std::literals;

#ifdef ADDGRAIN_X86
template<typename pixel_t, typename noise_t> extern void updateFrame_sse2(const void* _srcp, void* _dstp, const int width, const int height, const ptrdiff_t stride, const int noisePlane, const int noiseOffs, const AddGrainData* const VS_RESTRICT d) noexcept;
template<typename pixel_t, typename noise_t> extern void updateFrame_avx2(const void* _srcp, void* _dstp, const int width, const int height, const ptrdiff_t stride, const int noisePlane, const int noiseOffs, const AddGrainData* const VS_RESTRICT d) noexcept;
template<typename pixel_t, typename noise_t> extern void updateFrame_avx512(const void* _srcp, void* _dstp, const int width, const int height, const ptrdiff_t stride, const int noisePlane, const int noiseOffs, const AddGrainData* const VS_RESTRICT d) noexcept;
#endif

template<typename T>
static T getArg(const VSAPI* vsapi, const VSMap* map, const char* key, const T defaultValue) noexcept {
    T arg{};
    int err{};

    if constexpr (std::is_same_v<T, bool>)
        arg = !!vsapi->mapGetInt(map, key, 0, &err);
    else if constexpr (std::is_same_v<T, int> || std::is_same_v<T, long>)
        arg = vsapi->mapGetIntSaturated(map, key, 0, &err);
    else if constexpr (std::is_same_v<T, int64_t>)
        arg = vsapi->mapGetInt(map, key, 0, &err);
    else if constexpr (std::is_same_v<T, float>)
        arg = vsapi->mapGetFloatSaturated(map, key, 0, &err);
    else if constexpr (std::is_same_v<T, double>)
        arg = vsapi->mapGetFloat(map, key, 0, &err);

    if (err)
        arg = defaultValue;

    return arg;
}

static inline long fastUniformRandL(long& idum) noexcept {
    return idum = 1664525L * idum + 1013904223L;
}

// very fast & reasonably random
static inline float fastUniformRandF(long& idum) noexcept {
    // work with 32-bit IEEE floating point only!
    fastUniformRandL(idum);
    unsigned long itemp = 0x3f800000 | (0x007fffff & idum);
    return *reinterpret_cast<float*>(&itemp) - 1.0f;
}

static inline float gaussianRand(bool& iset, float& gset, long& idum) noexcept {
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

static inline float gaussianRand(const float mean, const float variance, bool& iset, float& gset, long& idum) noexcept {
    return (variance == 0.0f) ? mean : gaussianRand(iset, gset, idum) * std::sqrt(variance) + mean;
}

template<typename noise_t>
static void generateNoise(const int planesNoise, const float scale, AddGrainData* const VS_RESTRICT d) noexcept {
    float nRep[MAXP];
    for (auto i{ 0 }; i < MAXP; i++)
        nRep[i] = d->constant ? 1.0f : 2.0f;

    std::vector<float> lastLine(d->nStride[0]);
    constexpr auto mean{ 0.0f };
    const float pvar[]{ d->var, d->uvar };
    bool iset{};
    float gset{};
    auto pns{ d->pNoiseSeeds.begin() };

    for (auto plane{ 0 }; plane < planesNoise; plane++) {
        auto h{ static_cast<int>(std::ceil(d->nHeight[plane] * nRep[plane])) };
        if (planesNoise == 2 && plane == 1) {
            // fake plane needs at least one more row, and more if the rows are too small. round to the upper number
            h += (OFFSET_FAKEPLANE + d->nStride[plane] - 1) / d->nStride[plane];
        }
        d->nSize[plane] = d->nStride[plane] * h;

        // allocate space for noise
        d->pN[plane] = malloc(d->nSize[plane] * sizeof(noise_t));

        for (auto x{ 0 }; x < d->nStride[plane]; x++)
            lastLine[x] = gaussianRand(mean, pvar[plane], iset, gset, d->idum); // things to vertically smooth against

        for (auto y{ 0 }; y < h; y++) {
            auto pNW{ reinterpret_cast<noise_t*>(d->pN[plane]) + d->nStride[plane] * y };
            auto lastr{ gaussianRand(mean, pvar[plane], iset, gset, d->idum) }; // something to horiz smooth against

            for (auto x{ 0 }; x < d->nStride[plane]; x++) {
                auto r{ gaussianRand(mean, pvar[plane], iset, gset, d->idum) };

                r = lastr * d->hcorr + r * (1.0f - d->hcorr); // horizontal correlation
                lastr = r;

                r = lastLine[x] * d->vcorr + r * (1.0f - d->vcorr); // vert corr
                lastLine[x] = r;

                // set noise block
                if constexpr (std::is_integral_v<noise_t>)
                    *pNW++ = static_cast<noise_t>(std::round(r) * scale);
                else
                    *pNW++ = r * scale;
            }
        }

        for (auto x{ d->storedFrames }; x > 0; x--)
            *pns++ = fastUniformRandL(d->idum) & 0xff; // insert seed, to keep cache happy
    }
}

// on input, plane is the frame plane index (if applicable, 0 otherwise), and on output, it contains the selected noise plane
static void setRand(int& plane, int& noiseOffs, const int frameNumber, AddGrainData* const VS_RESTRICT d) noexcept {
    if (d->constant) {
        // force noise to be identical every frame
        if (plane >= MAXP) {
            plane = MAXP - 1;
            noiseOffs = OFFSET_FAKEPLANE;
        }
    } else {
        // pull seed back out, to keep cache happy
        auto seedIndex{ frameNumber % d->storedFrames };
        auto p0{ d->pNoiseSeeds[seedIndex] };

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
static void updateFrame_c(const void* _srcp, void* _dstp, const int width, const int height, const ptrdiff_t stride, const int noisePlane, const int noiseOffs,
                          const AddGrainData* const VS_RESTRICT d) noexcept {
    auto srcp{ reinterpret_cast<const pixel_t*>(_srcp) };
    auto dstp{ reinterpret_cast<pixel_t*>(_dstp) };
    auto pNW{ reinterpret_cast<noise_t*>(d->pN[noisePlane]) + noiseOffs };

    for (auto y{ 0 }; y < height; y++) {
        for (auto x{ 0 }; x < width; x++) {
            if constexpr (std::is_integral_v<pixel_t>)
                dstp[x] = static_cast<pixel_t>(std::clamp(srcp[x] + pNW[x], 0, d->peak));
            else
                dstp[x] = srcp[x] + pNW[x];
        }

        srcp += stride;
        dstp += stride;
        pNW += d->nStride[noisePlane];
    }
}

static const VSFrame* VS_CC addgrainGetFrame(int n, int activationReason, void* instanceData, [[maybe_unused]] void** frameData, VSFrameContext* frameCtx, VSCore* core, const VSAPI* vsapi) {
    auto d{ static_cast<AddGrainData*>(instanceData) };

    if (activationReason == arInitial) {
        vsapi->requestFrameFilter(n, d->node, frameCtx);
    } else if (activationReason == arAllFramesReady) {
        auto src{ vsapi->getFrameFilter(n, d->node, frameCtx) };
        decltype(src) fr[]{ d->process[0] ? nullptr : src, d->process[1] ? nullptr : src, d->process[2] ? nullptr : src };
        constexpr int pl[]{ 0, 1, 2 };
        auto dst{ vsapi->newVideoFrame2(&d->vi->format, d->vi->width, d->vi->height, fr, pl, src, core) };

        for (auto plane{ 0 }; plane < d->vi->format.numPlanes; plane++) {
            if (d->process[plane]) {
                const auto width{ vsapi->getFrameWidth(src, plane) };
                const auto height{ vsapi->getFrameHeight(src, plane) };
                const auto stride{ vsapi->getStride(src, plane) / d->vi->format.bytesPerSample };
                auto srcp{ vsapi->getReadPtr(src, plane) };
                auto dstp{ vsapi->getWritePtr(dst, plane) };

                auto noisePlane{ d->vi->format.colorFamily == cfRGB ? 0 : plane };
                auto noiseOffs{ 0 };
                setRand(noisePlane, noiseOffs, n, d); // seed randomness w/ plane & frame
                d->updateFrame(srcp, dstp, width, height, stride, noisePlane, noiseOffs, d);
            }
        }

        vsapi->freeFrame(src);
        return dst;
    }

    return nullptr;
}

static void VS_CC addgrainFree(void* instanceData, [[maybe_unused]] VSCore* core, const VSAPI* vsapi) {
    auto d{ static_cast<AddGrainData*>(instanceData) };

    vsapi->freeNode(d->node);

    for (auto i{ 0 }; i < MAXP; i++)
        free(d->pN[i]);

    delete d;
}

static void VS_CC addgrainCreate(const VSMap* in, VSMap* out, [[maybe_unused]] void* userData, VSCore* core, const VSAPI* vsapi) {
    auto d{ std::make_unique<AddGrainData>() };

    try {
        d->node = vsapi->mapGetNode(in, "clip", 0, nullptr);
        d->vi = vsapi->getVideoInfo(d->node);

        if (!vsh::isConstantVideoFormat(d->vi) ||
            (d->vi->format.sampleType == stInteger && d->vi->format.bitsPerSample > 16) ||
            (d->vi->format.sampleType == stFloat && d->vi->format.bitsPerSample != 32))
            throw "only constant format 8-16 bit integer and 32 bit float input supported";

        d->var = getArg(vsapi, in, "var", 1.0f);
        d->uvar = getArg(vsapi, in, "uvar", 0.0f);
        d->hcorr = getArg(vsapi, in, "hcorr", 0.0f);
        d->vcorr = getArg(vsapi, in, "vcorr", 0.0f);
        auto seed = getArg(vsapi, in, "seed", -1L);
        d->constant = getArg(vsapi, in, "constant", false);
        auto opt = getArg(vsapi, in, "opt", 0);

        if (d->hcorr < 0.0f || d->hcorr > 1.0f || d->vcorr < 0.0f || d->vcorr > 1.0f)
            throw "hcorr and vcorr must be between 0.0 and 1.0 (inclusive)";

        if (opt < 0 || opt > 4)
            throw "opt must be 0, 1, 2, 3, or 4";

        {
            if (d->vi->format.bytesPerSample == 1)
                d->updateFrame = updateFrame_c<uint8_t, int8_t>;
            else if (d->vi->format.bytesPerSample == 2)
                d->updateFrame = updateFrame_c<uint16_t, int16_t>;
            else
                d->updateFrame = updateFrame_c<float, float>;

#ifdef ADDGRAIN_X86
            auto iset{ instrset_detect() };
            if ((opt == 0 && iset >= 10) || opt == 4) {
                if (d->vi->format.bytesPerSample == 1) {
                    d->updateFrame = updateFrame_avx512<uint8_t, int8_t>;
                    d->step = 64;
                } else if (d->vi->format.bytesPerSample == 2) {
                    d->updateFrame = updateFrame_avx512<uint16_t, int16_t>;
                    d->step = 32;
                } else {
                    d->updateFrame = updateFrame_avx512<float, float>;
                    d->step = 16;
                }
            } else if ((opt == 0 && iset >= 8) || opt == 3) {
                if (d->vi->format.bytesPerSample == 1) {
                    d->updateFrame = updateFrame_avx2<uint8_t, int8_t>;
                    d->step = 32;
                } else if (d->vi->format.bytesPerSample == 2) {
                    d->updateFrame = updateFrame_avx2<uint16_t, int16_t>;
                    d->step = 16;
                } else {
                    d->updateFrame = updateFrame_avx2<float, float>;
                    d->step = 8;
                }
            } else if ((opt == 0 && iset >= 2) || opt == 2) {
                if (d->vi->format.bytesPerSample == 1) {
                    d->updateFrame = updateFrame_sse2<uint8_t, int8_t>;
                    d->step = 16;
                } else if (d->vi->format.bytesPerSample == 2) {
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
        if (d->vi->format.sampleType == stInteger) {
            scale = static_cast<float>(1 << (d->vi->format.bitsPerSample - 8));
            d->peak = (1 << d->vi->format.bitsPerSample) - 1;
        } else {
            scale = 1.0f / (d->vi->format.colorFamily == cfRGB ? 255.0f : 219.0f);
        }

        if (seed < 0)
            seed = static_cast<long>(std::time(nullptr)); // init random
        d->idum = seed;

        auto planesNoise{ 1 };
        d->nStride[0] = (d->vi->width + 63) & ~63;
        d->nHeight[0] = d->vi->height;
        if (d->vi->format.colorFamily == cfGray) {
            d->uvar = 0.0f;
        } else if (d->vi->format.colorFamily == cfRGB) {
            d->uvar = d->var;
        } else {
            planesNoise = 2;
            d->nStride[1] = ((d->vi->width >> d->vi->format.subSamplingW) + 63) & ~63;
            d->nHeight[1] = d->vi->height >> d->vi->format.subSamplingH;
        }

        if (d->var <= 0.0f && d->uvar <= 0.0f) {
            vsapi->mapConsumeNode(out, "clip", d->node, maReplace);
            return;
        }

        d->process[0] = d->var > 0.0f;
        d->process[1] = d->process[2] = d->uvar > 0.0f;

        d->storedFrames = std::min(d->vi->numFrames, 256);
        d->pNoiseSeeds.resize(d->storedFrames * planesNoise);

        if (d->vi->format.bytesPerSample == 1)
            generateNoise<int8_t>(planesNoise, scale, d.get());
        else if (d->vi->format.bytesPerSample == 2)
            generateNoise<int16_t>(planesNoise, scale, d.get());
        else
            generateNoise<float>(planesNoise, scale, d.get());
    } catch (const char* error) {
        vsapi->mapSetError(out, ("AddGrain: "s + error).c_str());
        vsapi->freeNode(d->node);
        return;
    }

    VSFilterDependency deps[]{ {d->node, rpStrictSpatial} };
    vsapi->createVideoFilter(out, "AddGrain", d->vi, addgrainGetFrame, addgrainFree, fmParallel, deps, 1, d.get(), core);
    d.release();
}

//////////////////////////////////////////
// Init

VS_EXTERNAL_API(void) VapourSynthPluginInit2(VSPlugin* plugin, const VSPLUGINAPI* vspapi) {
    vspapi->configPlugin("com.holywu.addgrain", "grain", "Random noise film grain generator", VS_MAKE_VERSION(10, 0), VAPOURSYNTH_API_VERSION, 0, plugin);
    vspapi->registerFunction("Add",
                             "clip:vnode;"
                             "var:float:opt;"
                             "uvar:float:opt;"
                             "hcorr:float:opt;"
                             "vcorr:float:opt;"
                             "seed:int:opt;"
                             "constant:int:opt;"
                             "opt:int:opt;",
                             "clip:vnode;",
                             addgrainCreate, nullptr, plugin);
}
