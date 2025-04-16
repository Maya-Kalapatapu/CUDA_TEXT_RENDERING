#include "cuda_text_wrapper.hpp"
#include "font_loader.hpp"
#include <cstdint>
#include <glm/vec4.hpp>
#include <iostream>  // ✅ Needed for std::cerr and std::cout

namespace portable_doc {

cuda::cuda_text cuda_text_wrapper::text_renderer;

cuda_text_wrapper::cuda_text_wrapper() {}
cuda_text_wrapper::~cuda_text_wrapper() {}

void cuda_text_wrapper::init() {
    text_renderer.init(816, 1056);  // Default A4 size
}

void cuda_text_wrapper::draw_page(const portable_doc& doc,
                                  const page& page,
                                  const section_style& style,
                                  const RenderSettings& settings) {
    // ✅ Safe conversion of doc text to string
    std::vector<char> raw_text = doc.get_text();
    std::string full_text;

    if (!raw_text.empty()) {
        full_text.assign(raw_text.begin(), raw_text.end());
    } else {
        std::cerr << "[Warning] portable_doc::get_text() returned empty content.\n";
        full_text = " ";
    }

    const line& l = doc.get_line(page.line_index);
    const format& fmt = doc.get_format(l.format_index);
    const font& f = doc.get_font(fmt.font_index);

    config.font_size = settings.font_size;

    // Calculate layout from section + render settings
    float spacing = fmt.line_height;
    int line_height = static_cast<int>(settings.font_size * spacing);
    int usable_height = style.page_height - style.margins.top_margin - style.margins.bottom_margin;
    int lines_per_page = usable_height / line_height;

    text_renderer.cleanup();
    text_renderer.draw_text(full_text, page.line_index, lines_per_page,
                            settings.text_color, settings.bg_color);
}

void cuda_text_wrapper::cleanup() {
    text_renderer.cleanup();
}

unsigned char* cuda_text_wrapper::get_bitmap() const {
    return text_renderer.get_bitmap();
}

GlyphInfo* cuda_text_wrapper::get_glyphs() const {
    return text_renderer.get_glyphs();
}

size_t cuda_text_wrapper::get_glyph_count() const {
    return text_renderer.get_glyph_count();
}

}
