#pragma once
#include <cuda_runtime.h>
#include "glyph_info.hpp"

void launch_text_kernel(uchar4* framebuffer, int width, int height,
                        unsigned char* glyph_bitmaps,
                        GlyphInfo* glyphs, int glyph_count,
                        uchar4 text_color, uchar4 bg_color);
