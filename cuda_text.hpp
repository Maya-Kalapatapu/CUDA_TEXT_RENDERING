#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <cuda_runtime.h>
#include "font_loader.hpp"
#include "glyph_info.hpp"

namespace cuda {

class cuda_text {
public:
    cuda_text();
    ~cuda_text();

    void init(int width, int height);
    void draw_text(const std::string& text,
                   int page_index,
                   int max_lines,
                   uchar4 text_color,
                   uchar4 bg_color);
    void cleanup();

    unsigned char* get_bitmap() const;
    GlyphInfo* get_glyphs() const;
    size_t get_glyph_count() const;

private:
    struct RenderData {
        std::vector<unsigned char> flat_bitmap;
        std::vector<GlyphInfo> glyphs;
    } render_data;

    struct CachedPage {
        unsigned char* d_bitmap = nullptr;
        GlyphInfo* d_glyphs = nullptr;
        size_t glyph_count = 0;
    };

    void cleanup_page(CachedPage& page);

    std::unordered_map<int, CachedPage> page_cache;

    unsigned char* d_bitmap = nullptr;
    GlyphInfo* d_glyphs = nullptr;
    int screen_width = 0;
    int screen_height = 0;
};

} // namespace cuda
