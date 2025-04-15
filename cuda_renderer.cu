#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "cuda_renderer.cuh"

__global__ void text_kernel(uchar4* pixels, int screen_w, int screen_h, unsigned char* glyph, int glyph_w, int glyph_h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= screen_w || y >= screen_h) return;

    int screen_idx = y * screen_w + x;
    pixels[screen_idx] = make_uchar4(0, 0, 0, 255);  // clear

    int gx = x - 100;
    int gy = y - 100;

    if (gx >= 0 && gx < glyph_w && gy >= 0 && gy < glyph_h) {
        int glyph_idx = gy * glyph_w + gx;
        unsigned char value = glyph[glyph_idx];
        pixels[screen_idx] = make_uchar4(value, value, value, 255);  // grayscale "A"
    }
}

void launch_text_kernel(uchar4* pixels, int width, int height, unsigned char* glyph, int gw, int gh) {
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    text_kernel<<<grid, block>>>(pixels, width, height, glyph, gw, gh);
}

