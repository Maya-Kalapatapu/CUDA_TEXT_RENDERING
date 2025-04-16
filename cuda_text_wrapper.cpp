#include "cuda_text_wrapper.hpp"
#include "font_loader.hpp"
#include <cstdint>
#include <glm/vec4.hpp>
#include <iostream>

namespace portable_doc {

cuda::cuda_text cuda_text_wrapper::text_renderer;

cuda_text_wrapper::cuda_text_wrapper() {}
cuda_text_wrapper::~cuda_text_wrapper() {}

void cuda_text_wrapper::init() {
    text_renderer.init(816, 1056);
}

void cuda_text_wrapper::draw_page(const portable_doc& doc,
                                  const page& pg,
                                  const section_style& style,
                                  const RenderSettings& settings,
                                  int start_line_index) {
    std::vector<char> raw_text = doc.get_text();
    std::string full_text;

    if (!raw_text.empty()) {
        full_text.assign(raw_text.begin(), raw_text.end());
    } else {
        std::cerr << "[Warning] portable_doc::get_text() returned empty content.\n";
        full_text = " ";
    }

    if (doc.get_num_pages() == 0 || doc.get_text().empty()) return;

    auto line = doc.get_line(0);
    auto fmt = doc.get_format(line.format_index);
    auto fnt = doc.get_font(fmt.font_index);

    config.font_size = settings.font_size;

    float spacing = fmt.line_height;
    int line_height = static_cast<int>(settings.font_size * spacing);
    int usable_height = style.page_height - style.margins.top_margin - style.margins.bottom_margin;
    int lines_per_page = usable_height / line_height;

    text_renderer.draw_text(full_text, start_line_index, lines_per_page,
                            settings.text_color, settings.bg_color);
}

void cuda_text_wrapper::preload_all_pages(const portable_doc& doc,
                                          const section_style& style,
                                          const RenderSettings& settings) {
    std::cout << "🚀 Preloading " << doc.get_num_pages() << " pages...\n";

    for (int i = 0; i < doc.get_num_pages(); ++i) {
        const auto& page = doc.get_page(i);

        if (i % 100 == 0) {
            std::cout << "   ↪️ Page " << i << "...\n";
        }

        draw_page(doc, page, style, settings, page.line_index);
    }

    std::cout << "✅ Preload complete.\n";
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
