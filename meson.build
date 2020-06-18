project('AddGrain', 'cpp',
  default_options: ['buildtype=release', 'b_ndebug=if-release', 'cpp_std=c++17'],
  meson_version: '>=0.48.0',
  version: '8'
)

sources = [
  'AddGrain/AddGrain.cpp',
  'AddGrain/AddGrain.h'
]

vapoursynth_dep = dependency('vapoursynth').partial_dependency(compile_args: true, includes: true)

libs = []

if host_machine.cpu_family().startswith('x86')
  add_project_arguments('-DADDGRAIN_X86', '-mfpmath=sse', '-msse2', language: 'cpp')

  sources += [
    'AddGrain/AddGrain_SSE2.cpp',
    'AddGrain/VCL2/instrset.h',
    'AddGrain/VCL2/instrset_detect.cpp',
    'AddGrain/VCL2/vector_convert.h',
    'AddGrain/VCL2/vectorclass.h',
    'AddGrain/VCL2/vectorf128.h',
    'AddGrain/VCL2/vectorf256.h',
    'AddGrain/VCL2/vectorf256e.h',
    'AddGrain/VCL2/vectorf512.h',
    'AddGrain/VCL2/vectorf512e.h',
    'AddGrain/VCL2/vectori128.h',
    'AddGrain/VCL2/vectori256.h',
    'AddGrain/VCL2/vectori256e.h',
    'AddGrain/VCL2/vectori512.h',
    'AddGrain/VCL2/vectori512e.h',
    'AddGrain/VCL2/vectori512s.h',
    'AddGrain/VCL2/vectori512se.h',
    'AddGrain/VCL2/vectormath_common.h',
    'AddGrain/VCL2/vectormath_exp.h',
    'AddGrain/VCL2/vectormath_hyp.h',
    'AddGrain/VCL2/vectormath_lib.h',
    'AddGrain/VCL2/vectormath_trig.h'
  ]

  libs += static_library('avx2', 'AddGrain/AddGrain_AVX2.cpp',
    dependencies: vapoursynth_dep,
    cpp_args: ['-mavx2', '-mfma'],
    gnu_symbol_visibility: 'hidden'
  )

  libs += static_library('avx512', 'AddGrain/AddGrain_AVX512.cpp',
    dependencies: vapoursynth_dep,
    cpp_args: ['-mavx512f', '-mavx512bw', '-mavx512dq', '-mavx512vl', '-mfma'],
    gnu_symbol_visibility: 'hidden'
  )
endif

shared_module('addgrain', sources,
  dependencies: vapoursynth_dep,
  link_with: libs,
  install: true,
  install_dir: join_paths(vapoursynth_dep.get_pkgconfig_variable('libdir'), 'vapoursynth'),
  gnu_symbol_visibility: 'hidden'
)
