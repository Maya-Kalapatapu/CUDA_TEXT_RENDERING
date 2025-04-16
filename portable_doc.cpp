#include "portable_doc.hpp"
#include <fstream>
#include <iostream>
#include <cstring>
#include <sstream>


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

uint32_t portable_doc::get_section_index(uint32_t) const {
    return 0;  // One section for now
}

void portable_doc::convert_text_to_pages(const std::string& text, uint32_t format_index) {
    set_text(text);

    formats.push_back({0, 1.2f, 0});
    fonts.push_back({16, "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"});
    colors.push_back({1.0f, 1.0f, 1.0f, 1.0f});
    section_styles.push_back({816, 1056, 0, {100, 100, 100, 100}});

    std::istringstream stream(text);
    std::string line;
    uint32_t offset = 0;

    // 🔁 Add one line per real text line
    while (std::getline(stream, line)) {
        lines.push_back({offset, format_index});
        offset += line.length() + 1;  // +1 for newline
    }

    // 🔁 Add one page per 50 lines
    for (uint32_t i = 0; i < lines.size(); i += 50) {
        pages.push_back({i, 0});
    }

    std::cout << "✅ Generated " << pages.size() << " pages from " << lines.size() << " lines.\n";
}

void portable_doc::save(const char* filename) const {
    std::ofstream out(filename, std::ios::binary);
    if (!out) {
        std::cerr << "Failed to open for saving: " << filename << std::endl;
        return;
    }

    auto write_vec = [&](auto& vec) {
        uint32_t size = static_cast<uint32_t>(vec.size());
        out.write(reinterpret_cast<const char*>(&size), sizeof(uint32_t));
        if (size > 0) {
            out.write(reinterpret_cast<const char*>(vec.data()), size * sizeof(decltype(vec[0])));
        }
    };

    uint32_t text_size = static_cast<uint32_t>(chars.size());
    out.write(reinterpret_cast<const char*>(&text_size), sizeof(uint32_t));
    out.write(reinterpret_cast<const char*>(chars.data()), text_size);

    write_vec(lines);
    write_vec(formats);
    write_vec(fonts);
    write_vec(colors);
    write_vec(section_styles);
    write_vec(pages);

    out.close();
}

void portable_doc::load(const char* filename) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) {
        std::cerr << "Failed to open .pdoc file: " << filename << std::endl;
        return;
    }

    chars.clear(); lines.clear(); formats.clear(); fonts.clear();
    colors.clear(); section_styles.clear(); pages.clear();

    uint32_t text_size = 0;
    in.read(reinterpret_cast<char*>(&text_size), sizeof(uint32_t));
    chars.resize(text_size);
    in.read(reinterpret_cast<char*>(chars.data()), text_size);

    auto read_vec = [&](auto& vec) {
        uint32_t size = 0;
        in.read(reinterpret_cast<char*>(&size), sizeof(uint32_t));
        vec.resize(size);
        if (size > 0) {
            in.read(reinterpret_cast<char*>(vec.data()), size * sizeof(decltype(vec[0])));
        }
    };

    read_vec(lines);
    read_vec(formats);
    read_vec(fonts);
    read_vec(colors);
    read_vec(section_styles);
    read_vec(pages);

    in.close();
}

}
