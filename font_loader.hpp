#pragma once
#include <string>
#include <map>
#include <vector>

struct Glyph {
    int width;
    int height;
    int pitch;
    int bearingX;
    int bearingY;
    int advance;
    std::vector<unsigned char> bitmap;
};

using GlyphAtlas = std::map<char, Glyph>;

// ✅ Configurable rendering parameters
struct TextRenderConfig {
    int font_size = 24;
};

extern TextRenderConfig config;

bool load_glyphs(const char* fontPath, const std::string& text, GlyphAtlas& atlas);
