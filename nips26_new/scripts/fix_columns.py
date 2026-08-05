import re

with open("main.tex", "r") as f:
    content = f.read()

# Tab 1: Line 328
content = content.replace(r"\begin{tabular}{l c c c c c c c c c}", r"\begin{tabular}{l c c c c c c c c >{\columncolor{blue!5}}c}")

# Tab 2: Line 447
content = content.replace(r"\begin{tabular}{c c c}", r"\begin{tabular}{c c >{\columncolor{blue!5}}c}")

# Tab 3: Line 479
content = content.replace(r"\begin{tabular}{c c c c c c c c c}", r"\begin{tabular}{c c c c c c c c >{\columncolor{blue!5}}c}")

# Tab 4: Line 775 (Ours(InstructPix2Pix) & Ours(ControlNet))
content = content.replace(r"\begin{tabular}{c c c c c c c c c | c}", r"\begin{tabular}{c c c c c c c c >{\columncolor{blue!5}}c >{\columncolor{blue!5}}c}")
content = content.replace(r"\begin{tabular}{c c c c c c c c c | c}", r"\begin{tabular}{c c c c c c c c >{\columncolor{blue!5}}c >{\columncolor{blue!5}}c}") # for line 796 too

# Appendix Tables
content = content.replace(r"\begin{tabular}{c c c c c c c c c}", r"\begin{tabular}{c c c c c c c c >{\columncolor{blue!5}}c}")

with open("main.tex", "w") as f:
    f.write(content)
