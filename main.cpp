#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "font_loader.hpp"
#include "cuda_renderer.cuh"
#include "font_loader.hpp"
#include <cuda_runtime.h>

unsigned char* d_glyph_bitmap = nullptr;
int glyph_w = 0, glyph_h = 0;

GLuint pbo, tex;
cudaGraphicsResource* cuda_pbo_resource;

const int WIDTH = 800;
const int HEIGHT = 600;

void render() {
    uchar4* dptr;
    size_t size;
    cudaGraphicsMapResources(1, &cuda_pbo_resource, 0);
    cudaGraphicsResourceGetMappedPointer((void**)&dptr, &size, cuda_pbo_resource);

    launch_text_kernel(dptr, WIDTH, HEIGHT, d_glyph_bitmap, glyph_w, glyph_h);

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

    GlyphBitmap glyph;
    if (!load_glyph_bitmap('A', glyph)) {
        std::cerr << "Failed to load glyph\n";
        return -1;
    }

    std::cout << "Loaded glyph 'A': "
            << glyph.width << "x" << glyph.height
            << ", pitch: " << glyph.pitch << std::endl;

    glyph_w = glyph.width;
    glyph_h = glyph.height;

    cudaMalloc(&d_glyph_bitmap, glyph_w * glyph_h);
    cudaMemcpy(d_glyph_bitmap, glyph.buffer, glyph_w * glyph_h, cudaMemcpyHostToDevice);

    free_glyph_bitmap(glyph);

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
