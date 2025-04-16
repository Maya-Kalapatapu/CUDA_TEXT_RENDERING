#include "cuda_text_wrapper.hpp"
#include "font_loader.hpp"
#include <cstdint>
#include <glm/vec4.hpp>  // for glm::vec4

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
                                  uchar4 text_color,
                                  uchar4 bg_color) {
    std::string full_text;
    for (uint32_t i = 0; i < doc.get_num_chars(); i++) {
        full_text += doc.get_text()[i];
    }

    int top = style.margins.top_margin;
    int bottom = style.margins.bottom_margin;
    int height = style.page_height;

    // Get format + font + color from the line
    const line& l = doc.get_line(page.line_index);
    const format& fmt = doc.get_format(l.format_index);
    const font& f = doc.get_font(fmt.font_index);
    glm::vec4 color = doc.get_color(fmt.color_index);

    config.font_size = f.get_size();

    // Override text color using format.color_index
    text_color = make_uchar4(
        static_cast<unsigned char>(color.r * 255),
        static_cast<unsigned char>(color.g * 255),
        static_cast<unsigned char>(color.b * 255),
        static_cast<unsigned char>(color.a * 255)
    );

    float spacing = fmt.line_height;
    int line_height = static_cast<int>(config.font_size * spacing);
    int usable_height = height - top - bottom;
    int lines_per_page = usable_height / line_height;

    text_renderer.cleanup();
    text_renderer.draw_text(full_text, page.line_index, lines_per_page, text_color, bg_color);
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
