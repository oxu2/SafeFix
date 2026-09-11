import re

with open("main.tex", "r") as f:
    content = f.read()

if r"\usepackage{arydshln}" not in content:
    content = content.replace(r"\usepackage{algorithm}", "\\usepackage{algorithm}\n\\usepackage{arydshln}\n")

with open("main.tex", "w") as f:
    f.write(content)

