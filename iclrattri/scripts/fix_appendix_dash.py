import re

with open("main.tex", "r") as f:
    content = f.read()

# For lines starting with "1.0 & ", insert "\cdashline{1-9}" before them
content = re.sub(r'\n1\.0 & ', '\n\\\\cdashline{1-9}\n1.0 & ', content)

with open("main.tex", "w") as f:
    f.write(content)
