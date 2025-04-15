#include "font_loader.hpp"
#include <ft2build.h>
#include FT_FREETYPE_H
#include <iostream>
#include <cstring>

static FT_Library ft;
static FT_Face face;

bool load_glyph_bitmap(char character, GlyphBitmap& out) {
    static bool initialized = false;

    if (!initialized) {
        if (FT_Init_FreeType(&ft)) {
            std::cerr << "Could not init FreeType library\n";
            return false;
        }

        const char* fontPath = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf";
        if (FT_New_Face(ft, fontPath, 0, &face)) {
            std::cerr << "Could not open " << fontPath << "\n";
            return false;
        }

        FT_Set_Pixel_Sizes(face, 0, 48);  // Set font size to 48px
        initialized = true;
    }

    if (FT_Load_Char(face, character, FT_LOAD_RENDER)) {
        std::cerr << "Could not load character '" << character << "'\n";
        return false;
    }

    out.width = face->glyph->bitmap.width;
    out.height = face->glyph->bitmap.rows;
    out.pitch = face->glyph->bitmap.pitch;
    out.buffer = new unsigned char[out.width * out.height];
    std::memcpy(out.buffer, face->glyph->bitmap.buffer, out.width * out.height);
    

    return true;
}

void free_glyph_bitmap(GlyphBitmap& bmp) {
    delete[] bmp.buffer;
}
