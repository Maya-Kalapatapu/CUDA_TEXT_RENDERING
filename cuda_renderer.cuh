#pragma once
#include <cuda_runtime.h>

void launch_text_kernel(uchar4* pixels, int width, int height, unsigned char* glyph, int gw, int gh);
