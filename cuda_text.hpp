#pragma once
#include <string>
#include <vector>
#include "font_loader.hpp"
#include "cuda_renderer.cuh"

namespace cuda {

class cuda_text {
public:
    cuda_text();
    ~cuda_text();

    void init(int width, int height);
    void draw_text(const std::string& text,
                   int start_line_index,
                   int max_lines,
                   uchar4 text_color,
                   uchar4 bg_color);
    void cleanup();

    unsigned char* get_bitmap() const;
    GlyphInfo* get_glyphs() const;
    size_t get_glyph_count() const;

private:
    int screen_width;
    int screen_height;

    unsigned char* d_bitmap = nullptr;
    GlyphInfo* d_glyphs = nullptr;

    struct RenderData {
        std::vector<unsigned char> flat_bitmap;
        std::vector<GlyphInfo> glyphs;
    } render_data;
};

}
