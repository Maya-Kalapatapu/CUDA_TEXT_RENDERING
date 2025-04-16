#pragma once
#include <cstdint>
#include <cuda_runtime.h>

struct RenderSettings {
    int font_size = 16;
    float line_spacing = 1.2f;
    uchar4 text_color = make_uchar4(255, 255, 255, 255);  // white
    uchar4 bg_color = make_uchar4(0, 0, 0, 255);          // black
};
