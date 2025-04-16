#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <iomanip>

#include "portable_doc.hpp"
#include "cuda_text_wrapper.hpp"
#include "render_settings.hpp"

const int PAGE_WIDTH = 816;
const int PAGE_HEIGHT = 1056;

// OpenGL-style benchmarking function
void print_speed(std::chrono::time_point<std::chrono::high_resolution_clock> t0, int start_page, int end_page) {
    auto t1 = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.0;
    int num_pages = end_page - start_page + 1;
    std::cout << std::fixed << std::setprecision(3)
              << ms << " ms to render " << num_pages << " pages >> "
              << (ms / num_pages) << " ms per page\n";
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: ./benchmark_render <file.txt or file.pdoc>\n";
        return 1;
    }

    std::string filepath = argv[1];
    portable_doc::portable_doc doc;

    // Auto-convert .txt to .pdoc
    if (filepath.size() >= 4 && filepath.substr(filepath.size() - 4) == ".txt") {
        std::ifstream in(filepath);
        std::stringstream buffer;
        buffer << in.rdbuf();
        std::string content = buffer.str();

        if (content.empty()) {
            std::cerr << "❌ Failed to read content from: " << filepath << "\n";
            return 1;
        }

        doc.set_text(content);
        doc.convert_text_to_pages(content, 0);
        doc.save("documents/converted_from_txt.pdoc");
        std::cout << "✅ Converted and saved: documents/converted_from_txt.pdoc\n";
    } else {
        doc.load(filepath.c_str());
    }

    int total_pages = std::min(10000u, doc.get_num_pages());
    std::cout << "Rendering " << total_pages << " pages...\n";

    if (total_pages == 0) {
        std::cerr << "❌ Document has 0 pages to render.\n";
        return 1;
    }

    // Setup CUDA renderer
    portable_doc::cuda_text_wrapper renderer;
    renderer.init();

    RenderSettings settings;
    settings.font_size = 16;
    settings.line_spacing = 1.2f;
    settings.text_color = make_uchar4(255, 255, 255, 255);
    settings.bg_color = make_uchar4(0, 0, 0, 255);

    auto t0 = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < total_pages; ++i) {
        auto page = doc.get_page(i);
        auto style = doc.get_section_style(doc.get_section_index(i));

        renderer.cleanup();
        renderer.draw_page(doc, page, style, settings);
    }

    print_speed(t0, 0, total_pages - 1);

    return 0;
}
