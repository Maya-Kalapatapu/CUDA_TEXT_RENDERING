#include "cuda_text.hpp"
#include "font_loader.hpp"
#include "cuda_renderer.cuh"
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <vector>
#include <string>
#include <iostream>
#include <sstream>
#include <algorithm>

namespace cuda {

cuda_text::cuda_text() {}
cuda_text::~cuda_text() { cleanup(); }

void cuda_text::init(int width, int height) {
    screen_width = width;
    screen_height = height;
}

void cuda_text::draw_text(const std::string& text, int start_line_index, int max_lines) {
    // ✅ Clear layout data
    render_data.flat_bitmap.clear();
    render_data.glyphs.clear();

    // Load all required glyphs
    GlyphAtlas atlas;
    if (!load_glyphs("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", text, atlas)) {
        std::cerr << "Failed to load glyphs\n";
        return;
    }

    // Measure glyph size for consistent spacing
    int max_glyph_height = 0;
    for (const auto& [ch, glyph] : atlas) {
        max_glyph_height = std::max(max_glyph_height, glyph.height);
    }

    float line_spacing_factor = 1.2f;
    int line_height = static_cast<int>(max_glyph_height * line_spacing_factor);

    // Set up margins and initial Y position
    int margin_left = 100;
    int margin_top = 100;
    int line_y = margin_top;

    std::istringstream stream(text);
    std::string line;
    std::vector<std::string> lines;

    while (std::getline(stream, line)) {
        lines.push_back(line);
    }

    int line_index = 0;
    for (int i = start_line_index; i < static_cast<int>(lines.size()) && line_index < max_lines; ++i, ++line_index) {
        int cursor_x = margin_left;
        const std::string& current_line = lines[i];

        for (char c : current_line) {
            if (!atlas.count(c)) continue;
            const Glyph& g = atlas[c];

            GlyphInfo info;
            info.x = cursor_x + g.bearingX;
            info.y = line_y + max_glyph_height - g.bearingY;  // ✅ Correct alignment
            info.width = g.width;
            info.height = g.height;
            info.bitmap_offset = render_data.flat_bitmap.size();

            render_data.flat_bitmap.insert(render_data.flat_bitmap.end(), g.bitmap.begin(), g.bitmap.end());
            render_data.glyphs.push_back(info);

            cursor_x += g.advance;
        }

        line_y += line_height;
    }

    // Upload to GPU
    cudaMalloc(&d_bitmap, render_data.flat_bitmap.size());
    cudaMemcpy(d_bitmap, render_data.flat_bitmap.data(), render_data.flat_bitmap.size(), cudaMemcpyHostToDevice);

    cudaMalloc(&d_glyphs, render_data.glyphs.size() * sizeof(GlyphInfo));
    cudaMemcpy(d_glyphs, render_data.glyphs.data(), render_data.glyphs.size() * sizeof(GlyphInfo), cudaMemcpyHostToDevice);
}

void cuda_text::cleanup() {
    if (d_bitmap) cudaFree(d_bitmap);
    if (d_glyphs) cudaFree(d_glyphs);
}

unsigned char* cuda_text::get_bitmap() const {
    return d_bitmap;
}

GlyphInfo* cuda_text::get_glyphs() const {
    return d_glyphs;
}

size_t cuda_text::get_glyph_count() const {
    return render_data.glyphs.size();
}

} // namespace cuda
