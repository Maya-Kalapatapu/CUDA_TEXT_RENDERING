#pragma once
#include <string>
#include <vector>
#include "cuda_renderer.cuh"

namespace cuda {

    class cuda_text {
        public:
            cuda_text();
            ~cuda_text();
        
            void init(int width, int height);
            void draw_text(const std::string& text);
            void cleanup();
        
            // ✅ Getter methods
            unsigned char* get_bitmap() const;
            GlyphInfo* get_glyphs() const;
            size_t get_glyph_count() const;
        
        private:
            unsigned char* d_bitmap = nullptr;
            GlyphInfo* d_glyphs = nullptr;
        
            int screen_width = 800;
            int screen_height = 600;
        
            struct RenderData {
                std::vector<unsigned char> flat_bitmap;
                std::vector<GlyphInfo> glyphs;
            } render_data;
        };
}        
