#pragma once
#include <cstdint>
#include <vector>
#include <map>
#include <string>

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

bool load_glyphs(const char* fontPath, const std::string& text, GlyphAtlas& atlas);
