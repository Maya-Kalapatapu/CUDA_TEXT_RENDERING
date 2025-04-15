import random

lorem = (
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
    "Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua."
)

with open("documents/sample_5_pages.txt", "w") as f:
    for i in range(250):
        line = f"{i+1:03d}: {lorem}\n"
        f.write(line)
