#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <vector>
#include <map>
#include <string>

#include "font_loader.hpp"
#include "cuda_renderer.cuh"

struct RenderData {
    std::vector<unsigned char> flat_bitmap;
    std::vector<GlyphInfo> glyphs;
};

unsigned char* d_bitmap = nullptr;
GlyphInfo* d_glyphs = nullptr;
RenderData renderData;

GLuint pbo, tex;
cudaGraphicsResource* cuda_pbo_resource;

const int WIDTH = 800;
const int HEIGHT = 600;

void render() {
    uchar4* dptr;
    size_t size;
    cudaGraphicsMapResources(1, &cuda_pbo_resource, 0);
    cudaGraphicsResourceGetMappedPointer((void**)&dptr, &size, cuda_pbo_resource);

    uchar4 text_color = make_uchar4(0, 255, 0, 255);  // Green text
    uchar4 bg_color   = make_uchar4(30, 30, 30, 255); // Dark gray background

    launch_text_kernel(dptr, WIDTH, HEIGHT, d_bitmap, d_glyphs, renderData.glyphs.size(), text_color, bg_color);

    cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0);
}

void display() {
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, WIDTH, HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE, 0);
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
    GLFWwindow* window = glfwCreateWindow(WIDTH, HEIGHT, "CUDA Font Renderer", NULL, NULL);
    if (!window) { glfwTerminate(); return -1; }
    glfwMakeContextCurrent(window);
    glewInit();

    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, WIDTH * HEIGHT * 4, 0, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo, cudaGraphicsMapFlagsWriteDiscard);

    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, WIDTH, HEIGHT, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);

    // Load all glyphs for the string
    GlyphAtlas atlas;
    std::string text = "Hello, CUDA!";
    const char* fontPath = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf";

    if (!load_glyphs(fontPath, text, atlas)) {
        std::cerr << "Failed to load glyphs\n";
        return -1;
    }

    int cursor_x = 100;  // Starting X
    int cursor_y = 300;  // Baseline Y

    for (char c : text) {
        if (!atlas.count(c)) continue;
        const Glyph& g = atlas[c];

        GlyphInfo info;
        info.x = cursor_x + g.bearingX;
        info.y = cursor_y - g.bearingY;
        info.width = g.width;
        info.height = g.height;
        info.bitmap_offset = renderData.flat_bitmap.size();

        renderData.flat_bitmap.insert(
            renderData.flat_bitmap.end(),
            g.bitmap.begin(),
            g.bitmap.end()
        );

        renderData.glyphs.push_back(info);
        cursor_x += g.advance;
    }

    // Upload glyph bitmap and metadata to CUDA
    cudaMalloc(&d_bitmap, renderData.flat_bitmap.size());
    cudaMemcpy(d_bitmap, renderData.flat_bitmap.data(), renderData.flat_bitmap.size(), cudaMemcpyHostToDevice);

    cudaMalloc(&d_glyphs, renderData.glyphs.size() * sizeof(GlyphInfo));
    cudaMemcpy(d_glyphs, renderData.glyphs.data(), renderData.glyphs.size() * sizeof(GlyphInfo), cudaMemcpyHostToDevice);

    while (!glfwWindowShouldClose(window)) {
        render();
        display();
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    cudaGraphicsUnregisterResource(cuda_pbo_resource);
    glDeleteBuffers(1, &pbo);
    glDeleteTextures(1, &tex);
    glfwTerminate();
    return 0;
}
