#pragma once

#include "portable_doc.hpp"
#include "cuda_text.hpp"

namespace portable_doc {

class cuda_text_wrapper {
public:
    cuda_text_wrapper();
    ~cuda_text_wrapper();

    void init();
    
    // ✅ Corrected: no portable_doc:: prefix on these
    void draw_page(const portable_doc& doc,
                   const page& page,
                   const section_style& style,
                   uchar4 text_color,
                   uchar4 bg_color);

    void cleanup();

    unsigned char* get_bitmap() const;
    GlyphInfo* get_glyphs() const;
    size_t get_glyph_count() const;

private:
    static cuda::cuda_text text_renderer;
};

}
