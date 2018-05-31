warning_flags = -Wall -Wextra -Wshadow -Wno-unused-parameter
common_cflags = -O3 -ffast-math -fvisibility=hidden $(warning_flags) $(MFLAGS)
AM_CXXFLAGS = -std=c++14 $(common_cflags)

AM_CPPFLAGS = $(VapourSynth_CFLAGS)

lib_LTLIBRARIES = libaddgrain.la

libaddgrain_la_SOURCES = AddGrain/AddGrain.cpp

libaddgrain_la_LDFLAGS = -no-undefined -avoid-version $(PLUGINLDFLAGS)
