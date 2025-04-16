#pragma once

#include "portable_doc.hpp"
#include "render_settings.hpp"
#include "cuda_text.hpp"

namespace portable_doc {

class cuda_text_wrapper {
public:
    cuda_text_wrapper();
    ~cuda_text_wrapper();

    void init();
    void cleanup();

    void draw_page(const portable_doc& doc,
                   const page& pg,
                   const section_style& style,
                   const RenderSettings& settings,
                   int start_line_index);

    unsigned char* get_bitmap() const;
    GlyphInfo* get_glyphs() const;
    size_t get_glyph_count() const;

private:
    static cuda::cuda_text text_renderer;
};

}
