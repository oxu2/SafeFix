import re

with open("main.tex", "r") as f:
    content = f.read()

# Replace \cdashline{2-10} with vertical space before and after
content = re.sub(r'\\cdashline\{2-10\}', r'\\noalign{\\vskip 1.5pt}\\cdashline{2-10}\\noalign{\\vskip 1.5pt}', content)

# Replace \cdashline{1-9} with vertical space before and after
content = re.sub(r'\\cdashline\{1-9\}', r'\\noalign{\\vskip 1.5pt}\\cdashline{1-9}\\noalign{\\vskip 1.5pt}', content)

with open("main.tex", "w") as f:
    f.write(content)
