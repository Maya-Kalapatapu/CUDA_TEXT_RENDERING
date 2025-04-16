#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include "portable_doc.hpp"
#include "cuda_text_wrapper.hpp"
#include "font_loader.hpp"
#include "cuda_renderer.cuh"
#include "render_settings.hpp"

const int PAGE_WIDTH = 816;
const int PAGE_HEIGHT = 1056;

GLuint pbo, tex;
cudaGraphicsResource* cuda_pbo_resource;

unsigned char* d_bitmap = nullptr;
GlyphInfo* d_glyphs = nullptr;
int glyph_count = 0;

std::string load_file(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        std::cerr << "Failed to open file: " << path << std::endl;
        return "";
    }
    std::stringstream buffer;
    buffer << in.rdbuf();
    return buffer.str();
}

void render(const RenderSettings& settings) {
    uchar4* dptr;
    size_t size;
    cudaGraphicsMapResources(1, &cuda_pbo_resource, 0);
    cudaGraphicsResourceGetMappedPointer((void**)&dptr, &size, cuda_pbo_resource);

    if (!d_bitmap || !d_glyphs || glyph_count == 0) {
        std::cerr << "⚠️ Skipping render due to invalid GPU buffers\n";
    } else {
        launch_text_kernel(dptr, PAGE_WIDTH, PAGE_HEIGHT, d_bitmap, d_glyphs, glyph_count,
                           settings.text_color, settings.bg_color);
    }

    cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0);
}

void display() {
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, PAGE_WIDTH, PAGE_HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    glClear(GL_COLOR_BUFFER_BIT);
    glEnable(GL_TEXTURE_2D);
    glBegin(GL_QUADS);
        glTexCoord2f(0, 1); glVertex2f(-1, -1);
        glTexCoord2f(1, 1); glVertex2f(1, -1);
        glTexCoord2f(1, 0); glVertex2f(1, 1);
        glTexCoord2f(0, 0); glVertex2f(-1, 1);
    glEnd();
}

int main(int argc, char** argv) {
    if (!glfwInit()) return -1;
    GLFWwindow* window = glfwCreateWindow(PAGE_WIDTH, PAGE_HEIGHT, "CUDA PortableDoc Renderer", NULL, NULL);
    if (!window) { glfwTerminate(); return -1; }
    glfwMakeContextCurrent(window);
    glewInit();

    RenderSettings settings;
    settings.font_size = 16;
    settings.line_spacing = 1.2f;
    settings.text_color = make_uchar4(255, 255, 255, 255);
    settings.bg_color = make_uchar4(0, 0, 0, 255);

    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, PAGE_WIDTH * PAGE_HEIGHT * 4, 0, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo, cudaGraphicsMapFlagsWriteDiscard);

    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, PAGE_WIDTH, PAGE_HEIGHT, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);

    std::string filepath = "documents/sample_numbered.txt";
    if (argc > 1) filepath = argv[1];

    portable_doc::portable_doc doc;

    if (filepath.size() >= 5 && filepath.substr(filepath.size() - 5) == ".pdoc") {
        doc.load(filepath.c_str());
    } else {
        std::string content = load_file(filepath);
        if (content.empty()) content = "Fallback content.";
        doc.set_text(content);
        doc.convert_text_to_pages(content, 0);
        doc.save("documents/generated.pdoc");
        std::cout << "✅ Saved .pdoc: documents/generated.pdoc\n";
    }

    portable_doc::cuda_text_wrapper renderer;
    renderer.init();

    auto style = doc.get_section_style(0);

    // ✅ Preload all pages up front
    renderer.preload_all_pages(doc, style, settings);

    uint32_t current_page = 0;

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS) {
            if (current_page + 1 < doc.get_num_pages()) current_page++;
            glfwWaitEventsTimeout(0.2);
        }

        if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS) {
            if (current_page > 0) current_page--;
            glfwWaitEventsTimeout(0.2);
        }

        const auto& page = doc.get_page(current_page);
        renderer.cleanup();  // Optional: remove if cache should persist between scrolls
        renderer.draw_page(doc, page, style, settings, page.line_index);

        d_bitmap = renderer.get_bitmap();
        d_glyphs = renderer.get_glyphs();
        glyph_count = renderer.get_glyph_count();

        render(settings);
        display();
        glfwSwapBuffers(window);
    }

    cudaGraphicsUnregisterResource(cuda_pbo_resource);
    glDeleteBuffers(1, &pbo);
    glDeleteTextures(1, &tex);
    glfwTerminate();
    return 0;
}
