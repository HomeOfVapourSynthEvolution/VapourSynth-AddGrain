// VapourSynth port by HolyWu
//
// Copyright (c) 2002 Tom Barry.  All rights reserved.
//      trbarry@trbarry.com
//  modified by Foxyshadis
//      foxyshadis@hotmail.com
//  modified by Firesledge
//      http://ldesoras.free.fr
//  modified by LaTo INV.
//      http://forum.doom9.org/member.php?u=131032
// Requires Avisynth source code to compile for Avisynth
// Avisynth Copyright 2000 Ben Rudiak-Gould.
//      http://www.math.berkeley.edu/~benrg/avisynth.html
/////////////////////////////////////////////////////////////////////////////
//
//  This file is subject to the terms of the GNU General Public License as
//  published by the Free Software Foundation.  A copy of this license is
//  included with this software distribution in the file COPYING.  If you
//  do not have a copy, you may obtain a copy by writing to the Free
//  Software Foundation, 675 Mass Ave, Cambridge, MA 02139, USA.
//
//  This software is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details
//  
//  Also, this program is "Philanthropy-Ware".  That is, if you like it and 
//  feel the need to reward or inspire the author then please feel free (but
//  not obligated) to consider joining or donating to the Electronic Frontier
//  Foundation. This will help keep cyber space free of barbed wire and bullsh*t.  
//
/////////////////////////////////////////////////////////////////////////////
// Change Log
//
// Date          Version  Developer      Changes
//
// 07 May 2003   1.0.0.0  Tom Barry      New Release
// 01 Jun 2006   1.1.0.0  Foxyshadis     Chroma noise, constant seed
// 06 Jun 2006   1.2.0.0  Foxyshadis     Supports YUY2, RGB. Fix cache mess.
// 10 Jun 2006   1.3.0.0  Foxyshadis     Crashfix, noisegen optimization
// 11 Nov 2006   1.4.0.0  Foxyshadis     Constant replaces seed, seed repeatable
// 07 May 2010   1.5.0.0  Foxyshadis     Limit the initial seed generation to fix memory issues.
// 13 May 2010   1.5.1.0  Firesledge     The source code compiles on Visual C++ versions older than 2008
// 26 Oct 2011   1.5.2.0  Firesledge     Removed the SSE2 requirement.
// 26 Oct 2011   1.5.3.0  Firesledge     Fixed coloring and bluring in RGB24 mode.
// 27 Oct 2011   1.5.4.0  Firesledge     Fixed bad pixels on the last line in YV12 mode when constant=true,
//                                       fixed potential problems with frame width > 4096 pixels
//                                       and fixed several other minor things.
// 28 Oct 2011   1.6.0.0  LaTo INV.      Added SSE2 code (50% faster than MMX).
// 29 Oct 2011   1.6.1.0  LaTo INV.      Automatic switch to MMX if SSE2 is not supported by the CPU.
// 16 Aug 2012   1.7.0.0  Firesledge     Supports Y8, YV16, YV24 and YV411 colorspaces.
//
/////////////////////////////////////////////////////////////////////////////

#include <algorithm>
#include <cassert>
#include <cmath>
#include <ctime>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include <VapourSynth.h>
#include <VSHelper.h>

// max # of noise planes
static constexpr int MAXP = 2;

// offset in pixels of the fake plane MAXP relative to plane MAXP-1
static constexpr int OFFSET_FAKEPLANE = 32;

struct AddGrainData {
    VSNodeRef * node;
    const VSVideoInfo * vi;
    bool constant;
    int64_t idum;
    int nStride[MAXP], nHeight[MAXP], nSize[MAXP], storedFrames;
    std::vector<uint8_t> pNoiseSeeds;
    std::vector<int16_t> pN[MAXP];
    std::vector<float> pNF[MAXP];
    bool process[3];
};

static inline int64_t fastUniformRandL(int64_t * idum) noexcept {
    return *idum = 1664525LL * (*idum) + 1013904223LL;
}

// very fast & reasonably random
static inline float fastUniformRandF(int64_t * idum) noexcept {
    // work with 32-bit IEEE floating point only!
    fastUniformRandL(idum);
    const uint64_t itemp = 0x3f800000 | (0x007fffff & *idum);
    return *reinterpret_cast<const float *>(&itemp) - 1.f;
}

static inline float gaussianRand(bool * iset, float * gset, int64_t * idum) noexcept {
    float fac, rsq, v1, v2;

    // return saved second
    if (*iset) {
        *iset = false;
        return *gset;
    }

    do {
        v1 = 2.f * fastUniformRandF(idum) - 1.f;
        v2 = 2.f * fastUniformRandF(idum) - 1.f;
        rsq = v1 * v1 + v2 * v2;
    } while (rsq >= 1.f || rsq == 0.f);

    fac = std::sqrt(-2.f * std::log(rsq) / rsq);

    // function generates two values every iteration, so save one for later
    *gset = v1 * fac;
    *iset = true;

    return v2 * fac;
}

static inline float gaussianRand(const float mean, const float variance, bool * iset, float * gset, int64_t * idum) noexcept {
    return (variance == 0.f) ? mean : gaussianRand(iset, gset, idum) * std::sqrt(variance) + mean;
}

// on input, plane is the frame plane index (if applicable, 0 otherwise), and on output, it contains the selected noise plane
static void setRand(int * plane, int * noiseOffs, const int frameNumber, AddGrainData * d) {
    if (d->constant) {
        // force noise to be identical every frame
        if (*plane >= MAXP) {
            *plane = MAXP - 1;
            *noiseOffs = OFFSET_FAKEPLANE;
        }
    } else {
        // pull seed back out, to keep cache happy
        const int seedIndex = frameNumber % d->storedFrames;
        const int p0 = d->pNoiseSeeds[seedIndex];

        if (*plane == 0) {
            d->idum = p0;
        } else {
            d->idum = d->pNoiseSeeds[seedIndex + d->storedFrames];
            if (*plane == 2) {
                // the trick to needing only 2 planes ^.~
                d->idum ^= p0;
                (*plane)--;
            }
        }

        // start noise at random qword in top half of noise area
        *noiseOffs = static_cast<int>(fastUniformRandF(&d->idum) * d->nSize[*plane] / MAXP) & 0xfffffff8;
    }

    assert(*plane >= 0);
    assert(*plane < MAXP);
    assert(*noiseOffs >= 0);
    assert(*noiseOffs < d->nSize[*plane]); // minimal check
}

template<typename T1, typename T2 = void>
static void updateFrame(T1 * VS_RESTRICT dstp, const int width, const int height, const int stride, const int noisePlane, const int noiseOffs, const AddGrainData * const VS_RESTRICT d) {
    const int shift1 = (sizeof(T1) == sizeof(uint8_t)) ? 0 : 16 - d->vi->format->bitsPerSample;
    constexpr int shift2 = (sizeof(T1) == sizeof(uint8_t)) ? 8 : 0;
    constexpr int lower = std::numeric_limits<T2>::min();
    constexpr int upper = std::numeric_limits<T2>::max();

    const int16_t * pNW = d->pN[noisePlane].data() + noiseOffs;
    assert(noiseOffs + d->nStride[noisePlane] * (height - 1) + stride <= d->nSize[noisePlane]);

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            T2 val = (dstp[x] << shift1) ^ lower;
            const T2 nz = pNW[x] >> shift2;
            val = std::min(std::max(val + nz, lower), upper);
            dstp[x] = val ^ lower;
            dstp[x] >>= shift1;
        }

        dstp += stride;
        pNW += d->nStride[noisePlane];
    }
}

template<>
void updateFrame(float * VS_RESTRICT dstp, const int width, const int height, const int stride, const int noisePlane, const int noiseOffs, const AddGrainData * const VS_RESTRICT d) {
    const float * pNW = d->pNF[noisePlane].data() + noiseOffs;
    assert(noiseOffs + d->nStride[noisePlane] * (height - 1) + stride <= d->nSize[noisePlane]);

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++)
            dstp[x] += pNW[x];

        dstp += stride;
        pNW += d->nStride[noisePlane];
    }
}

static void VS_CC addgrainInit(VSMap *in, VSMap *out, void **instanceData, VSNode *node, VSCore *core, const VSAPI *vsapi) {
    AddGrainData * d = static_cast<AddGrainData *>(*instanceData);
    vsapi->setVideoInfo(d->vi, 1, node);
}

static const VSFrameRef *VS_CC addgrainGetFrame(int n, int activationReason, void **instanceData, void **frameData, VSFrameContext *frameCtx, VSCore *core, const VSAPI *vsapi) {
    AddGrainData * d = static_cast<AddGrainData *>(*instanceData);

    if (activationReason == arInitial) {
        vsapi->requestFrameFilter(n, d->node, frameCtx);
    } else if (activationReason == arAllFramesReady) {
        const VSFrameRef * src = vsapi->getFrameFilter(n, d->node, frameCtx);
        VSFrameRef * dst = vsapi->copyFrame(src, core);

        for (int plane = 0; plane < d->vi->format->numPlanes; plane++) {
            if (d->process[plane]) {
                const int width = vsapi->getFrameWidth(dst, plane);
                const int height = vsapi->getFrameHeight(dst, plane);
                const int stride = vsapi->getStride(dst, plane);
                uint8_t * dstp = vsapi->getWritePtr(dst, plane);

                int noisePlane = (d->vi->format->colorFamily == cmRGB) ? 0 : plane;
                int noiseOffs = 0;
                setRand(&noisePlane, &noiseOffs, n, d); // seeds randomness w/ plane & frame

                if (d->vi->format->bytesPerSample == 1)
                    updateFrame<uint8_t, int8_t>(dstp, width, height, stride, noisePlane, noiseOffs, d);
                else if (d->vi->format->bytesPerSample == 2)
                    updateFrame<uint16_t, int16_t>(reinterpret_cast<uint16_t *>(dstp), width, height, stride / 2, noisePlane, noiseOffs, d);
                else
                    updateFrame<float>(reinterpret_cast<float *>(dstp), width, height, stride / 4, noisePlane, noiseOffs, d);
            }
        }

        vsapi->freeFrame(src);
        return dst;
    }

    return nullptr;
}

static void VS_CC addgrainFree(void *instanceData, VSCore *core, const VSAPI *vsapi) {
    AddGrainData * d = static_cast<AddGrainData *>(instanceData);
    vsapi->freeNode(d->node);
    delete d;
}

static void VS_CC addgrainCreate(const VSMap *in, VSMap *out, void *userData, VSCore *core, const VSAPI *vsapi) {
    std::unique_ptr<AddGrainData> d = std::make_unique<AddGrainData>();
    int err;

    d->node = vsapi->propGetNode(in, "clip", 0, nullptr);
    d->vi = vsapi->getVideoInfo(d->node);

    try {
        if (!isConstantFormat(d->vi) || (d->vi->format->sampleType == stInteger && d->vi->format->bitsPerSample > 16) ||
            (d->vi->format->sampleType == stFloat && d->vi->format->bitsPerSample != 32))
            throw std::string{ "only constant format 8-16 bit integer and 32 bit float input supported" };

        float var = static_cast<float>(vsapi->propGetFloat(in, "var", 0, &err));
        if (err)
            var = 1.f;

        float uvar = static_cast<float>(vsapi->propGetFloat(in, "uvar", 0, &err));

        const float hcorr = static_cast<float>(vsapi->propGetFloat(in, "hcorr", 0, &err));

        const float vcorr = static_cast<float>(vsapi->propGetFloat(in, "vcorr", 0, &err));

        int64_t seed = vsapi->propGetInt(in, "seed", 0, &err);
        if (err)
            seed = -1;

        d->constant = !!vsapi->propGetInt(in, "constant", 0, &err);

        if (hcorr < 0.f || hcorr > 1.f || vcorr < 0.f || vcorr > 1.f)
            throw std::string{ "hcorr and vcorr must be between 0.0 and 1.0 (inclusive)" };

        bool iset = false;
        float gset;

        if (seed < 0)
            seed = std::time(nullptr); // init random
        d->idum = seed;

        int planesNoise = 1;
        d->nStride[0] = (d->vi->width + 15) & ~15; // first plane
        d->nHeight[0] = d->vi->height;
        if (d->vi->format->colorFamily == cmGray) {
            uvar = 0.f;
        } else if (d->vi->format->colorFamily == cmRGB) {
            uvar = var;
        } else {
            planesNoise = 2;
            d->nStride[1] = ((d->vi->width >> d->vi->format->subSamplingW) + 15) & ~15; // second and third plane
            d->nHeight[1] = d->vi->height >> d->vi->format->subSamplingH;
        }

        if (var <= 0.f && uvar <= 0.f) {
            vsapi->propSetNode(out, "clip", d->node, paReplace);
            vsapi->freeNode(d->node);
            return;
        }

        d->storedFrames = std::min(d->vi->numFrames, 256);
        d->pNoiseSeeds.resize(d->storedFrames * planesNoise);
        auto pns = d->pNoiseSeeds.begin();

        float nRep[] = { 2.f, 2.f };
        if (d->constant)
            nRep[0] = nRep[1] = 1.f;

        const float pvar[] = { var, uvar };
        std::vector<float> lastLine(d->nStride[0]); // assume plane 0 is the widest one
        const float mean = 0.f;

        for (int plane = 0; plane < planesNoise; plane++) {
            int h = static_cast<int>(std::ceil(d->nHeight[plane] * nRep[plane]));
            if (planesNoise == 2 && plane == 1) {
                // fake plane needs at least one more row, and more if the rows are too small. round to the upper number
                h += (OFFSET_FAKEPLANE + d->nStride[plane] - 1) / d->nStride[plane];
            }
            d->nSize[plane] = d->nStride[plane] * h;

            // allocate space for noise
            if (d->vi->format->sampleType == stInteger)
                d->pN[plane].resize(d->nSize[plane]);
            else
                d->pNF[plane].resize(d->nSize[plane]);

            for (int x = 0; x < d->nStride[plane]; x++)
                lastLine[x] = gaussianRand(mean, pvar[plane], &iset, &gset, &d->idum); // things to vertically smooth against

            for (int y = 0; y < h; y++) {
                if (d->vi->format->sampleType == stInteger) {
                    auto pNW = d->pN[plane].begin() + d->nStride[plane] * y;
                    float lastr = gaussianRand(mean, pvar[plane], &iset, &gset, &d->idum); // something to horiz smooth against

                    for (int x = 0; x < d->nStride[plane]; x++) {
                        float r = gaussianRand(mean, pvar[plane], &iset, &gset, &d->idum);

                        r = lastr * hcorr + r * (1.f - hcorr); // horizontal correlation
                        lastr = r;

                        r = lastLine[x] * vcorr + r * (1.f - vcorr); // vert corr
                        lastLine[x] = r;

                        *pNW++ = static_cast<int16_t>(std::round(r * 256.f)); // set noise block
                    }
                } else {
                    auto pNW = d->pNF[plane].begin() + d->nStride[plane] * y;
                    float lastr = gaussianRand(mean, pvar[plane], &iset, &gset, &d->idum); // something to horiz smooth against

                    for (int x = 0; x < d->nStride[plane]; x++) {
                        float r = gaussianRand(mean, pvar[plane], &iset, &gset, &d->idum);

                        r = lastr * hcorr + r * (1.f - hcorr); // horizontal correlation
                        lastr = r;

                        r = lastLine[x] * vcorr + r * (1.f - vcorr); // vert corr
                        lastLine[x] = r;

                        *pNW++ = r / 255.f; // set noise block
                    }
                }
            }

            for (int x = d->storedFrames; x > 0; x--)
                *pns++ = fastUniformRandL(&d->idum) & 0xff; // insert seed, to keep cache happy
        }

        d->process[0] = var > 0.f;
        d->process[1] = d->process[2] = uvar > 0.f;
    } catch (const std::string & error) {
        vsapi->setError(out, ("AddGrain: " + error).c_str());
        vsapi->freeNode(d->node);
        return;
    }

    vsapi->createFilter(in, out, "AddGrain", addgrainInit, addgrainGetFrame, addgrainFree, fmParallelRequests, 0, d.release(), core);
}

//////////////////////////////////////////
// Init

VS_EXTERNAL_API(void) VapourSynthPluginInit(VSConfigPlugin configFunc, VSRegisterFunction registerFunc, VSPlugin *plugin) {
    configFunc("com.holywu.addgrain", "grain", "Add some correlated color gaussian noise", VAPOURSYNTH_API_VERSION, 1, plugin);
    registerFunc("Add",
                 "clip:clip;"
                 "var:float:opt;"
                 "uvar:float:opt;"
                 "hcorr:float:opt;"
                 "vcorr:float:opt;"
                 "seed:int:opt;"
                 "constant:int:opt;",
                 addgrainCreate, nullptr, plugin);
}
