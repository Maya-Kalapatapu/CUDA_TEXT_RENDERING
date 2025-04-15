#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "cuda_renderer.cuh"

__global__ void text_kernel(
    uchar4* framebuffer, int screen_w, int screen_h,
    unsigned char* bitmaps, GlyphInfo* glyphs, int glyph_count,
    uchar4 text_color, uchar4 bg_color
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= screen_w || y >= screen_h) return;

    int pixel_idx = y * screen_w + x;
    framebuffer[pixel_idx] = bg_color;

    for (int i = 0; i < glyph_count; ++i) {
        GlyphInfo g = glyphs[i];

        int gx = x - g.x;
        int gy = y - g.y;

        if (gx >= 0 && gx < g.width && gy >= 0 && gy < g.height) {
            int glyph_pixel_index = g.bitmap_offset + gy * g.width + gx;
            unsigned char alpha = bitmaps[glyph_pixel_index];

            if (alpha > 0) {
                framebuffer[pixel_idx] = make_uchar4(
                    (text_color.x * alpha) / 255,
                    (text_color.y * alpha) / 255,
                    (text_color.z * alpha) / 255,
                    255
                );
            }
        }
    }
}

void launch_text_kernel(uchar4* framebuffer, int width, int height,
                        unsigned char* glyph_bitmaps,
                        GlyphInfo* glyphs, int glyph_count,
                        uchar4 text_color, uchar4 bg_color) {
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    text_kernel<<<grid, block>>>(
        framebuffer, width, height,
        glyph_bitmaps, glyphs, glyph_count,
        text_color, bg_color
    );
}
