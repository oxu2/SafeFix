import re

with open("main.tex", "r") as f:
    content = f.read()

# 1. Remove \cellcolor{pink!50}
content = content.replace(r"\cellcolor{pink!50}", "")

# 2. Change columncolor{pink!50} to columncolor{blue!5}
content = content.replace(r"columncolor{pink!50}", r"columncolor{blue!5}")

# 3. Change \cmidrule{2-10} and \cmidrule{2-9} and \cmidrule{2-8} to \cdashline
content = content.replace(r"\cmidrule{2-10}", r"\cdashline{2-10}")
content = content.replace(r"\cmidrule{2-9}", r"\cdashline{2-9}")
content = content.replace(r"\cmidrule{2-8}", r"\cdashline{2-8}")

with open("main.tex", "w") as f:
    f.write(content)

