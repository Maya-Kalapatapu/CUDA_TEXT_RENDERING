#include "portable_doc.hpp"

namespace portable_doc {

portable_doc::portable_doc() {}

void portable_doc::set_text(const std::string& str) {
    chars.assign(str.begin(), str.end());
}

std::vector<char> portable_doc::get_text() const {
    return chars;
}

uint32_t portable_doc::get_num_chars() const {
    return static_cast<uint32_t>(chars.size());
}

line portable_doc::get_line(uint32_t index) const {
    return lines.at(index);
}

format portable_doc::get_format(uint32_t index) const {
    return formats.at(index);
}

font portable_doc::get_font(uint32_t index) const {
    return fonts.at(index);
}

glm::vec4 portable_doc::get_color(uint32_t index) const {
    return colors.at(index);
}

page portable_doc::get_page(uint32_t index) const {
    return pages.at(index);
}

uint32_t portable_doc::get_num_pages() const {
    return static_cast<uint32_t>(pages.size());
}

section_style portable_doc::get_section_style(uint32_t index) const {
    return section_styles.at(index);
}

uint32_t portable_doc::get_section_index(uint32_t page_index) const {
    return 0;  // Simplified: always return default section
}

void portable_doc::convert_text_to_pages(const std::string& text, uint32_t format_index) {
    set_text(text);

    // Create a default format + font + color
    formats.push_back({0, 1.2f, 0});
    fonts.push_back({16, "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"});
    colors.push_back({1.0f, 1.0f, 1.0f, 1.0f}); // white

    // One section style
    section_styles.push_back({816, 1056, 0, {100, 100, 100, 100}});

    // Create lines
    uint32_t index = 0;
    while (index < chars.size()) {
        lines.push_back({index, format_index});
        // Line break every ~80 chars (simulate wrapping)
        index += 80;
    }

    // Create pages (every 50 lines)
    for (uint32_t i = 0; i < lines.size(); i += 50) {
        pages.push_back({i, 0});
    }
}

}
