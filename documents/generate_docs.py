import os
#from reportlab.pdfgen import canvas

# Toggle this to generate the large test file
generate_massive_txt = True

os.makedirs("documents", exist_ok=True)

def write_file(path, lines):
    with open(path, "w", encoding="utf-8") as f:
        for i, line in enumerate(lines, 1):
            f.write(f"{i:06}: {line}\n")

# 1. Short doc
short_line = "This is a short test document line."
write_file("documents/sample_short.txt", [short_line] * 10)

# 2. Long doc (~5 pages)
long_lines = [f"This is line {i+1:03} of the long document for paging test." for i in range(250)]
write_file("documents/sample_long.txt", long_lines)

# 3. Wrapped doc
wrapped_line = ("This is a long sentence designed to exceed the width of the page and "
                "test how word wrapping works in the CUDA renderer.")
write_file("documents/sample_wrapped.txt", [wrapped_line] * 50)

# 4. Unicode doc
unicode_line = "Emoji & accents: café, résumé, naïve — 👍🔥✨"
write_file("documents/sample_unicode.txt", [unicode_line] * 20)

# 5. Simulated .pdoc file (text-only, for conversion testing)
write_file("documents/sample.pdoc", long_lines * 100)

# 7. 🔥 MASSIVE TEXT FILE (~10,000 pages)
if generate_massive_txt:
    print("⏳ Generating massive 10,000-page text file (~500,000 lines)...")
    mass_line = "This is a line in the massive test document."
    massive_lines = [mass_line] * 500_000
    write_file("documents/sample_massive_10000_pages.txt", massive_lines)

print("✅ Test documents generated in ./documents (TXT, PDC, PDF)")
