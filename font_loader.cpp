#include "font_loader.hpp"
#include <ft2build.h>
#include FT_FREETYPE_H
#include <iostream>
#include <cstring>

TextRenderConfig config;

bool load_glyphs(const char* fontPath, const std::string& text, GlyphAtlas& atlas) {
    FT_Library ft;
    if (FT_Init_FreeType(&ft)) {
        std::cerr << "Could not init FreeType\n";
        return false;
    }

    FT_Face face;
    if (FT_New_Face(ft, fontPath, 0, &face)) {
        std::cerr << "Could not open font\n";
        return false;
    }

    // ✅ Use configurable font size
    FT_Set_Pixel_Sizes(face, 0, config.font_size);

    for (char c : text) {
        if (atlas.count(c)) continue;

        if (FT_Load_Char(face, c, FT_LOAD_RENDER)) {
            std::cerr << "Failed to load glyph: " << c << "\n";
            continue;
        }

        Glyph g;
        g.width = face->glyph->bitmap.width;
        g.height = face->glyph->bitmap.rows;
        g.pitch = face->glyph->bitmap.pitch;
        g.bearingX = face->glyph->bitmap_left;
        g.bearingY = face->glyph->bitmap_top;
        g.advance = face->glyph->advance.x >> 6;

        size_t size = g.width * g.height;
        g.bitmap.resize(size);
        std::memcpy(g.bitmap.data(), face->glyph->bitmap.buffer, size);

        atlas[c] = g;
    }

    FT_Done_Face(face);
    FT_Done_FreeType(ft);
    return true;
}
