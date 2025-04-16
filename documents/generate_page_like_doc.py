lines = 200
line_length = 80

lorem = (
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
    "Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua."
)

with open("documents/sample_doc_a4.txt", "w") as f:
    for i in range(lines):
        line = f"{i+1:03d}: {lorem[:line_length]}\n"
        f.write(line)
