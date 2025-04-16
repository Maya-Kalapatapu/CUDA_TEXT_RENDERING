#include "cuda_text.hpp"
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <sstream>
#include <algorithm>
#include <iostream>

namespace cuda {

cuda_text::cuda_text() {}
cuda_text::~cuda_text() { cleanup(); }

void cuda_text::init(int width, int height) {
    screen_width = width;
    screen_height = height;
}

void cuda_text::draw_text(const std::string& text,
                          int start_line_index,
                          int max_lines,
                          uchar4 text_color,
                          uchar4 bg_color) {
    // ✅ Use start_line_index for caching
    if (page_cache.count(start_line_index)) {
        std::cout << "[CACHE HIT] line " << start_line_index << "\n";
        const CachedPage& cached = page_cache[start_line_index];
        d_bitmap = cached.d_bitmap;
        d_glyphs = cached.d_glyphs;
        return;
    }

    std::cout << "[CACHE MISS] line " << start_line_index << "\n";

    render_data.flat_bitmap.clear();
    render_data.glyphs.clear();

    GlyphAtlas atlas;
    if (!load_glyphs("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", text, atlas)) {
        std::cerr << "Failed to load glyphs\n";
        return;
    }

    int max_glyph_height = 0;
    for (const auto& [ch, glyph] : atlas) {
        max_glyph_height = std::max(max_glyph_height, glyph.height);
    }

    float spacing = 1.2f;
    int line_height = static_cast<int>(max_glyph_height * spacing);

    int margin_top = 100, margin_bottom = 100, margin_left = 100, margin_right = 100;
    int usable_height = screen_height - margin_top - margin_bottom;
    if (max_lines <= 0) max_lines = usable_height / line_height;

    int line_y = margin_top;

    std::istringstream stream(text);
    std::string paragraph;
    std::vector<std::string> all_lines;

    while (std::getline(stream, paragraph)) {
        std::istringstream wordstream(paragraph);
        std::string word, current_line;
        int line_width = 0;

        while (wordstream >> word) {
            int word_width = 0;
            for (char c : word) if (atlas.count(c)) word_width += atlas[c].advance;
            if (!current_line.empty()) word_width += atlas[' '].advance;

            if (line_width + word_width > screen_width - margin_left - margin_right) {
                all_lines.push_back(current_line);
                current_line = word;
                line_width = word_width;
            } else {
                if (!current_line.empty()) current_line += ' ';
                current_line += word;
                line_width += word_width;
            }
        }

        if (!current_line.empty()) all_lines.push_back(current_line);
    }

    int line_index = 0;
    for (int i = start_line_index; i < static_cast<int>(all_lines.size()) && line_index < max_lines; ++i, ++line_index) {
        const std::string& line = all_lines[i];
        int cursor_x = margin_left;

        for (char c : line) {
            if (!atlas.count(c)) continue;
            const Glyph& g = atlas[c];

            GlyphInfo info;
            info.x = cursor_x + g.bearingX;
            info.y = line_y + max_glyph_height - g.bearingY;
            info.width = g.width;
            info.height = g.height;
            info.bitmap_offset = render_data.flat_bitmap.size();

            render_data.flat_bitmap.insert(render_data.flat_bitmap.end(), g.bitmap.begin(), g.bitmap.end());
            render_data.glyphs.push_back(info);

            cursor_x += g.advance;
        }

        line_y += line_height;
    }

    CachedPage cached;

    cudaMalloc(&cached.d_bitmap, render_data.flat_bitmap.size());
    cudaMemcpy(cached.d_bitmap, render_data.flat_bitmap.data(), render_data.flat_bitmap.size(), cudaMemcpyHostToDevice);

    cudaMalloc(&cached.d_glyphs, render_data.glyphs.size() * sizeof(GlyphInfo));
    cudaMemcpy(cached.d_glyphs, render_data.glyphs.data(), render_data.glyphs.size() * sizeof(GlyphInfo), cudaMemcpyHostToDevice);

    cached.glyph_count = render_data.glyphs.size();
    page_cache[start_line_index] = cached;

    d_bitmap = cached.d_bitmap;
    d_glyphs = cached.d_glyphs;
}

void cuda_text::cleanup_page(CachedPage& page) {
    if (page.d_bitmap) {
        cudaFree(page.d_bitmap);
        page.d_bitmap = nullptr;
    }
    if (page.d_glyphs) {
        cudaFree(page.d_glyphs);
        page.d_glyphs = nullptr;
    }
}

void cuda_text::cleanup() {
    for (auto& [_, page] : page_cache) {
        cleanup_page(page);
    }
    page_cache.clear();
    d_bitmap = nullptr;
    d_glyphs = nullptr;
}

unsigned char* cuda_text::get_bitmap() const { return d_bitmap; }
GlyphInfo* cuda_text::get_glyphs() const { return d_glyphs; }
size_t cuda_text::get_glyph_count() const { return render_data.glyphs.size(); }

}
