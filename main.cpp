#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include "cuda_text.hpp"
#include "cuda_renderer.cuh"
#include "font_loader.hpp"
#include "page_manager.hpp"

const int PAGE_WIDTH = 816;
const int PAGE_HEIGHT = 1056;
const int MARGIN_TOP = 100;
const int MARGIN_BOTTOM = 100;

GLuint pbo, tex;
cudaGraphicsResource* cuda_pbo_resource;

unsigned char* d_bitmap = nullptr;
GlyphInfo* d_glyphs = nullptr;
int glyph_count = 0;

std::vector<std::string> split_into_lines(const std::string& text) {
    std::istringstream stream(text);
    std::string line;
    std::vector<std::string> lines;
    while (std::getline(stream, line)) {
        lines.push_back(line);
    }
    return lines;
}

std::string load_file(const std::string& path) {
    std::ifstream in(path);
    std::stringstream buffer;
    buffer << in.rdbuf();
    return buffer.str();
}

void render() {
    uchar4* dptr;
    size_t size;
    cudaGraphicsMapResources(1, &cuda_pbo_resource, 0);
    cudaGraphicsResourceGetMappedPointer((void**)&dptr, &size, cuda_pbo_resource);

    uchar4 text_color = make_uchar4(0, 255, 0, 255);
    uchar4 bg_color   = make_uchar4(30, 30, 30, 255);

    launch_text_kernel(dptr, PAGE_WIDTH, PAGE_HEIGHT, d_bitmap, d_glyphs, glyph_count, text_color, bg_color);

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

int main() {
    if (!glfwInit()) return -1;
    GLFWwindow* window = glfwCreateWindow(PAGE_WIDTH, PAGE_HEIGHT, "CUDA Text Renderer", NULL, NULL);
    if (!window) { glfwTerminate(); return -1; }
    glfwMakeContextCurrent(window);
    glewInit();

    // ✅ Set font size and calculate lines per page
    config.font_size = 20;
    float line_spacing_factor = 1.2f;
    int line_height = static_cast<int>(config.font_size * line_spacing_factor);
    int usable_height = PAGE_HEIGHT - MARGIN_TOP - MARGIN_BOTTOM;
    int lines_per_page = usable_height / line_height;

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

    std::string full_text = load_file("documents/sample_doc_a4.txt");
    std::vector<std::string> all_lines = split_into_lines(full_text);

    PageManager pager(all_lines.size(), lines_per_page);

    cuda::cuda_text text_renderer;
    text_renderer.init(PAGE_WIDTH, PAGE_HEIGHT);

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        bool ctrl = glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS ||
                    glfwGetKey(window, GLFW_KEY_RIGHT_CONTROL) == GLFW_PRESS;

        if (ctrl && glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS) {
            pager.next_page();
            glfwWaitEventsTimeout(0.2);
        }

        if (ctrl && glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS) {
            pager.prev_page();
            glfwWaitEventsTimeout(0.2);
        }

        int start_line_index = pager.get_start_line_index();

        text_renderer.cleanup();
        text_renderer.draw_text(full_text, start_line_index, lines_per_page);
        d_bitmap = text_renderer.get_bitmap();
        d_glyphs = text_renderer.get_glyphs();
        glyph_count = text_renderer.get_glyph_count();

        render();
        display();
        glfwSwapBuffers(window);
    }

    cudaGraphicsUnregisterResource(cuda_pbo_resource);
    glDeleteBuffers(1, &pbo);
    glDeleteTextures(1, &tex);
    glfwTerminate();
    return 0;
}
