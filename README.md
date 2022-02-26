# AddGrain
AddGrain generates film like grain or other effects (like rain) by adding random noise to a video clip. This noise may optionally be horizontally or vertically correlated to cause streaking.

Ported from AviSynth plugin http://forum.doom9.org/showthread.php?t=111849


## Usage
    grain.Add(vnode clip[, float var=1.0, float uvar=0.0, float hcorr=0.0, float vcorr=0.0, int seed=-1, bint constant=False, int opt=0])

- clip: Clip to process. Any format with either integer sample type of 8-16 bit depth or float sample type of 32 bit depth is supported.

- var, uvar: The variance (strength) of the luma and chroma noise, 0 is disabled. `uvar` does nothing for GRAY and RGB formats.

- hcorr, vcorr: Horizontal and vertical correlation, which causes a nifty streaking effect. Range 0.0-1.0.

- seed: Specifies a repeatable grain sequence. Set to at least 0 to use.

- constant: Specifies a constant grain pattern on every frame.

- opt: Sets which cpu optimizations to use.
  - 0 = auto detect
  - 1 = use c
  - 2 = use sse2
  - 3 = use avx2
  - 4 = use avx512

The correlation factors are actually just implemented as exponential smoothing which give a weird side affect that I did not attempt to adjust. But this means that as you increase either corr factor you will have to also increase the stddev (grain amount) in order to get the same visible amount of grain, since it is being smooth out a bit.

Increase both corr factors can somewhat give clumps, or larger grain size.

And there is an interesting effect with, say, `grain.Add(var=800, vcorr=0.9)` or any huge amount of strongly vertical grain. It can make a scene look like it is raining.


## Compilation
```
meson build
ninja -C build
ninja -C build install
```
