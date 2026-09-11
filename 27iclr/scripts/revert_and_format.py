import sys
import re

with open('main.tex', 'r') as f:
    content = f.read()

# 1. Revert Limitations
box_start = r"""\begin{tcolorbox}[
    colback=red!3!white,
    colframe=red!50!black,
    title=Limitations \& Practical Considerations,
    fonttitle=\bfseries,
    rounded corners,
    drop shadow,
    breakable
]
"""
content = content.replace(box_start, "")
content = content.replace("remains useful.\n\\end{tcolorbox}", "remains useful.")

# 2. Revert Appendix tables (only in appendix part, e.g., after \appendix or roughly line > 600)
# We can just target the exact strings that were generated for the appendix tables.
content = content.replace(r"""\renewcommand{\arraystretch}{1.1}
\begin{tabular}{c c c c c c c c >{\columncolor{pink!50}}c}
\toprule
\rowcolor{sectiongray} Ratio""", r"""\begin{tabular}{c c c c c c c c c}
\toprule
Ratio""")

content = content.replace(r"""\renewcommand{\arraystretch}{1.1}
\begin{tabular}{c c c c c c c c >{\columncolor{pink!50}}c | >{\columncolor{pink!50}}c}
\toprule
\rowcolor{sectiongray} Ratio""", r"""\begin{tabular}{c c c c c c c c c | c}
\toprule
Ratio""")

content = content.replace(r"""\renewcommand{\arraystretch}{1.1}
\begin{tabular}{p{0.16\linewidth} p{0.43\linewidth} p{0.31\linewidth}}
\toprule
\rowcolor{sectiongray} Category""", r"""\begin{tabular}{p{0.16\linewidth} p{0.43\linewidth} p{0.31\linewidth}}
\toprule
Category""")

# 3. Format Proposition and Proof
prop_box_start = r"""\begin{tcolorbox}[
    colback=blue!2!white,
    colframe=blue!20!white,
    arc=3pt,
    boxrule=1pt,
    drop shadow,
    breakable
]
\begin{proposition}"""
prop_box_end = r"""\end{proposition}
\end{tcolorbox}"""

content = content.replace(r'\begin{proposition}', prop_box_start)
content = content.replace(r'\end{proposition}', prop_box_end)

proof_box_start = r"""\begin{tcolorbox}[
    colback=green!2!white,
    colframe=green!20!white,
    arc=3pt,
    boxrule=1pt,
    drop shadow,
    breakable
]
\begin{proof}"""
proof_box_end = r"""\end{proof}
\end{tcolorbox}"""

content = content.replace(r'\begin{proof}', proof_box_start)
content = content.replace(r'\end{proof}', proof_box_end)


with open('main.tex', 'w') as f:
    f.write(content)
print("Updated main.tex")
