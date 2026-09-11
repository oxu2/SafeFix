import sys

with open('main.tex', 'r') as f:
    content = f.read()

# 1. Add color definitions
if '\\definecolor{sectiongray}' not in content:
    content = content.replace('\\usepackage{colortbl}       % colored tables\n', '\\usepackage{colortbl}       % colored tables\n\\definecolor{sectiongray}{gray}{0.90}\n\\definecolor{headergray}{gray}{0.96}\n')

# 2. Global replacements for the appendix tables we already touched
content = content.replace('\\rowcolor{gray!15}', '\\rowcolor{sectiongray}')
content = content.replace('columncolor{blue!5}', 'columncolor{pink!50}')

# 3. Main Text Table 1 (tab:main-rer)
t1_old = r"""\begin{tabular}{l c c c c c c c c c}
\toprule
\textbf{Dataset} & Ratio & Random & Stratified & Entropy & CoreSet & BADGE & UHerding & PPAL & \textbf{Ours} \\
\midrule"""
t1_new = r"""\renewcommand{\arraystretch}{1.1}
\begin{tabular}{l c c c c c c c c >{\columncolor{pink!50}}c}
\toprule
\rowcolor{sectiongray} \textbf{Dataset} & Ratio & Random & Stratified & Entropy & CoreSet & BADGE & UHerding & PPAL & \textbf{Ours} \\
\midrule"""
content = content.replace(t1_old, t1_new)

# 4. Main Text Table 2 (tab:selection-efficiency)
t2_old = r"""\begin{tabular}{l c c c c c c c c c}
\toprule
Dataset & Ratio & Random & Stratified & Entropy & CoreSet & BADGE & UHerding & PPAL & \textbf{Ours / speedup} \\
\midrule"""
t2_new = r"""\renewcommand{\arraystretch}{1.1}
\begin{tabular}{l c c c c c c c c >{\columncolor{pink!50}}c}
\toprule
\rowcolor{sectiongray} Dataset & Ratio & Random & Stratified & Entropy & CoreSet & BADGE & UHerding & PPAL & \textbf{Ours / speedup} \\
\midrule"""
content = content.replace(t2_old, t2_new)

# 5. Main Text Table 3 (tab:imagenet-human-audit)
t3_old = r"""\begin{tabular}{c c c c c c c c c}
\toprule
Ratio & Random & Stratified & Entropy & CoreSet & BADGE & UHerding & PPAL & Ours(ControlNet) \\
\midrule"""
t3_new = r"""\renewcommand{\arraystretch}{1.1}
\begin{tabular}{c c c c c c c c >{\columncolor{pink!50}}c}
\toprule
\rowcolor{sectiongray} Ratio & Random & Stratified & Entropy & CoreSet & BADGE & UHerding & PPAL & Ours(ControlNet) \\
\midrule"""
content = content.replace(t3_old, t3_new)

# 6. Add \renewcommand{\arraystretch}{1.1} to the appendix tables
app_t1_old = r"""\begin{tabular}{c c c c c c c c >{\columncolor{pink!50}}c}
\toprule
\rowcolor{sectiongray} Ratio"""
app_t1_new = r"""\renewcommand{\arraystretch}{1.1}
\begin{tabular}{c c c c c c c c >{\columncolor{pink!50}}c}
\toprule
\rowcolor{sectiongray} Ratio"""
content = content.replace(app_t1_old, app_t1_new)

app_t2_old = r"""\begin{tabular}{c c c c c c c c >{\columncolor{pink!50}}c | >{\columncolor{pink!50}}c}
\toprule
\rowcolor{sectiongray} Ratio"""
app_t2_new = r"""\renewcommand{\arraystretch}{1.1}
\begin{tabular}{c c c c c c c c >{\columncolor{pink!50}}c | >{\columncolor{pink!50}}c}
\toprule
\rowcolor{sectiongray} Ratio"""
content = content.replace(app_t2_old, app_t2_new)

app_t3_old = r"""\begin{tabular}{p{0.16\linewidth} p{0.43\linewidth} p{0.31\linewidth}}
\toprule
\rowcolor{sectiongray} Category"""
app_t3_new = r"""\renewcommand{\arraystretch}{1.1}
\begin{tabular}{p{0.16\linewidth} p{0.43\linewidth} p{0.31\linewidth}}
\toprule
\rowcolor{sectiongray} Category"""
content = content.replace(app_t3_old, app_t3_new)

with open('main.tex', 'w') as f:
    f.write(content)
print("Updated main.tex")
