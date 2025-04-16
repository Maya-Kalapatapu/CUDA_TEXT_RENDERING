#pragma once

class PageManager {
public:
    PageManager(int total_lines, int lines_per_page);

    void next_page();
    void prev_page();
    void goto_page(int page);
    
    int get_start_line_index() const;
    int get_lines_per_page() const;
    int get_current_page() const;
    int get_total_pages() const;

private:
    int total_lines;
    int lines_per_page;
    int page_index;
};
