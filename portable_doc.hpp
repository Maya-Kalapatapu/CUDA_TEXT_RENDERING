#pragma once

#include <vector>
#include <string>
#include <cstdint>
#include <glm/vec4.hpp>

namespace portable_doc {

struct margin {
    uint32_t top_margin = 100;
    uint32_t bottom_margin = 100;
    uint32_t left_margin = 100;
    uint32_t right_margin = 100;
};

struct section_style {
    uint32_t page_width;
    uint32_t page_height;
    uint32_t bg_color_index;
    margin margins;
};

struct page {
    uint32_t line_index;
    uint32_t view_index;
};

struct line {
    uint32_t text_index;
    uint32_t format_index;
};

struct format {
    uint32_t font_index;
    float line_height;
    uint32_t color_index;
};

struct font {
    uint32_t size;
    std::string path;

    uint32_t get_size() const { return size; }
};

class portable_doc {
public:
    portable_doc();

    void set_text(const std::string& str);
    std::vector<char> get_text() const;
    uint32_t get_num_chars() const;

    line get_line(uint32_t index) const;
    format get_format(uint32_t index) const;
    font get_font(uint32_t index) const;
    glm::vec4 get_color(uint32_t index) const;

    page get_page(uint32_t index) const;
    uint32_t get_num_pages() const;
    section_style get_section_style(uint32_t index) const;
    uint32_t get_section_index(uint32_t page_index) const;

    void convert_text_to_pages(const std::string& text, uint32_t format_index);

    // ✅ New
    void save(const char* filename) const;
    void load(const char* filename);

private:
    std::vector<char> chars;
    std::vector<line> lines;
    std::vector<format> formats;
    std::vector<font> fonts;
    std::vector<glm::vec4> colors;
    std::vector<page> pages;
    std::vector<section_style> section_styles;
};

}
