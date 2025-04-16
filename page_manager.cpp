#include "page_manager.hpp"
#include <algorithm>

PageManager::PageManager(int total_lines, int lines_per_page)
    : total_lines(total_lines),
      lines_per_page(lines_per_page),
      page_index(0) {}

void PageManager::next_page() {
    int max_page = (total_lines - 1) / lines_per_page;
    if (page_index < max_page) {
        page_index++;
    }
}

void PageManager::prev_page() {
    if (page_index > 0) {
        page_index--;
    }
}

void PageManager::goto_page(int page) {
    int max_page = (total_lines - 1) / lines_per_page;
    page_index = std::clamp(page, 0, max_page);
}

int PageManager::get_start_line_index() const {
    return page_index * lines_per_page;
}

int PageManager::get_lines_per_page() const {
    return lines_per_page;
}

int PageManager::get_current_page() const {
    return page_index + 1;
}

int PageManager::get_total_pages() const {
    return (total_lines - 1) / lines_per_page + 1;
}
