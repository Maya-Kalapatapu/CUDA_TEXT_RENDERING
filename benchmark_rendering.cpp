#include <chrono>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "portable_doc.hpp"
#include "cuda_text_wrapper.hpp"
#include "render_settings.hpp"

int main(int argc, char** argv) {
    std::string filepath = "documents/sample_numbered.txt";
    if (argc > 1) filepath = argv[1];

    portable_doc::portable_doc doc;

    if (filepath.size() >= 4 && filepath.substr(filepath.size() - 4) == ".txt") {
        std::ifstream in(filepath);
        std::stringstream buffer;
        buffer << in.rdbuf();
        std::string content = buffer.str();
        doc.set_text(content);
        doc.convert_text_to_pages(content, 0);
        doc.save("documents/converted_from_txt.pdoc");
        std::cout << "✅ Converted and saved: documents/converted_from_txt.pdoc\n";
    } else {
        doc.load(filepath.c_str());
    }

    int total_pages = std::min(10000u, doc.get_num_pages());
    std::cout << "🧪 Benchmarking " << total_pages << " pages...\n";

    if (total_pages == 0) {
        std::cerr << "❌ Document has 0 pages to render.\n";
        return 1;
    }

    portable_doc::cuda_text_wrapper renderer;
    renderer.init();

    RenderSettings settings;
    settings.font_size = 16;
    settings.line_spacing = 1.2f;
    settings.text_color = make_uchar4(255, 255, 255, 255);
    settings.bg_color = make_uchar4(0, 0, 0, 255);

    auto style = doc.get_section_style(0);

    // ✅ Preload and time it
    auto preload_start = std::chrono::steady_clock::now();
    renderer.preload_all_pages(doc, style, settings);
    auto preload_end = std::chrono::steady_clock::now();
    double preload_time = std::chrono::duration<double>(preload_end - preload_start).count() * 1000.0;
    std::cout << "🛠 Preload Time: " << preload_time << " ms\n";

    // Benchmark loop
    auto wall_start = std::chrono::steady_clock::now();

    for (int i = 0; i < total_pages; ++i) {
        const auto& page = doc.get_page(i);
        if (i % 100 == 0) std::cout << "Rendering page " << i << "/" << total_pages << "\n";
        renderer.draw_page(doc, page, style, settings, page.line_index);
    }

    auto wall_end = std::chrono::steady_clock::now();
    double wall_ms = std::chrono::duration<double>(wall_end - wall_start).count() * 1000.0;

    // Output
    std::cout << "\n========================\n";
    std::cout << "🕒 Total Time:          " << wall_ms << " ms\n";
    std::cout << "📄 Pages:               " << total_pages << "\n";
    std::cout << "🕒 Avg Time per Page:   " << (wall_ms / total_pages) << " ms\n";
    std::cout << "========================\n";

    std::cout << "\n🧠 Metric Definitions:\n";
    std::cout << "  Total Time (ms): Time from start to finish of the benchmark loop.\n";
    std::cout << "  Avg Time/Page:   Average total time spent rendering one page.\n";
    std::cout << "  ⚠ Note: This includes all CPU + GPU work — real-world rendering time.\n";

    return 0;
}
