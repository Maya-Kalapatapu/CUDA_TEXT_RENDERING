#pragma once
#include <cstdint>

struct GlyphBitmap {
    int width;
    int height;
    int pitch;
    unsigned char* buffer;
};

bool load_glyph_bitmap(char character, GlyphBitmap& out);
void free_glyph_bitmap(GlyphBitmap& bmp);
