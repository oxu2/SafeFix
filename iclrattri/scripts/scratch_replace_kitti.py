import re

with open("main.tex", "r") as f:
    content = f.read()

# Find the exact Table 5 block
table_block = r"""\begin{table}[h]
\centering
\caption{KITTI object-detection attribute corpus used for VLM assignment. The placeholder \texttt{\#1} is replaced by a candidate value from the corresponding category.}
\label{tab:kitti-attribute-corpus}
\scriptsize
\setlength{\tabcolsep}{2.5pt}
\renewcommand{\arraystretch}{1.08}
\begin{tabular}{p{0.16\linewidth} p{0.43\linewidth} p{0.31\linewidth}}
\toprule
Category & Candidate values & Prompt template \\
\midrule
\texttt{color} &
\texttt{white}, \texttt{black}, \texttt{gray}, \texttt{silver}, \texttt{red}, \texttt{blue}, \texttt{green}, \texttt{yellow}, \texttt{brown} &
\texttt{a \#1 main target object in a driving scene} \\
\texttt{object\_scale} &
\texttt{small}, \texttt{medium}, \texttt{large} &
\texttt{a \#1 target object in a driving scene} \\
\texttt{viewpoint} &
\texttt{front}, \texttt{rear}, \texttt{side}, \texttt{front-side}, \texttt{rear-side} &
\texttt{a \#1 view of the main target object in a driving scene} \\
\texttt{occlusion} &
\texttt{none}, \texttt{partial}, \texttt{heavy} &
\texttt{a \#1 occluded target object in a driving scene} \\
\texttt{truncation} &
\texttt{none}, \texttt{partial}, \texttt{heavy} &
\texttt{a \#1 truncated target object in a driving scene} \\
\texttt{motion\_state} &
\texttt{moving}, \texttt{stationary} &
\texttt{a \#1 target object in a driving scene} \\
\texttt{background\_style} &
\texttt{city street}, \texttt{residential street}, \texttt{highway}, \texttt{parking area}, \texttt{service lane}, \texttt{tree-lined road} &
\texttt{a driving scene on a \#1} \\
\texttt{traffic\_density} &
\texttt{sparse}, \texttt{light}, \texttt{moderate}, \texttt{heavy} &
\texttt{a driving scene with \#1 traffic} \\
\texttt{lighting\_condition} &
\texttt{bright}, \texttt{normal}, \texttt{dim}, \texttt{dark} &
\texttt{a driving scene under \#1 lighting condition} \\
\texttt{weather} &
\texttt{clear}, \texttt{cloudy}, \texttt{rainy}, \texttt{snowy}, \texttt{foggy} &
\texttt{a driving scene with \#1 weather} \\
\bottomrule
\end{tabular}
\end{table}"""

replacement = r"""\begin{table}[htbp]
\centering
\caption{KITTI object-detection attribute corpus used for VLM assignment.}
\label{tab:kitti-attribute-corpus}
\begin{tcolorbox}[
    colback=blue!2!white,
    colframe=blue!15!black,
    title=\textbf{KITTI Attribute Corpus \& VLM Prompts},
    fonttitle=\bfseries,
    rounded corners,
    drop shadow,
    breakable,
    left=6pt,
    right=6pt,
    top=4pt,
    bottom=4pt,
    fontupper=\small,
    width=0.98\linewidth
]
\textit{Note: The placeholder \texttt{\#1} is replaced by a candidate value from the corresponding category.}
\vspace{0.8em}

\begin{description}
    \setlength{\itemsep}{0.6em}
    \item[\texttt{color}:] \texttt{white}, \texttt{black}, \texttt{gray}, \texttt{silver}, \texttt{red}, \texttt{blue}, \texttt{green}, \texttt{yellow}, \texttt{brown} \\
    \textcolor{gray}{\textbf{Prompt:}} \texttt{a \#1 main target object in a driving scene}

    \item[\texttt{object\_scale}:] \texttt{small}, \texttt{medium}, \texttt{large} \\
    \textcolor{gray}{\textbf{Prompt:}} \texttt{a \#1 target object in a driving scene}

    \item[\texttt{viewpoint}:] \texttt{front}, \texttt{rear}, \texttt{side}, \texttt{front-side}, \texttt{rear-side} \\
    \textcolor{gray}{\textbf{Prompt:}} \texttt{a \#1 view of the main target object in a driving scene}

    \item[\texttt{occlusion}:] \texttt{none}, \texttt{partial}, \texttt{heavy} \\
    \textcolor{gray}{\textbf{Prompt:}} \texttt{a \#1 occluded target object in a driving scene}

    \item[\texttt{truncation}:] \texttt{none}, \texttt{partial}, \texttt{heavy} \\
    \textcolor{gray}{\textbf{Prompt:}} \texttt{a \#1 truncated target object in a driving scene}

    \item[\texttt{motion\_state}:] \texttt{moving}, \texttt{stationary} \\
    \textcolor{gray}{\textbf{Prompt:}} \texttt{a \#1 target object in a driving scene}

    \item[\texttt{background\_style}:] \texttt{city street}, \texttt{residential street}, \texttt{highway}, \texttt{parking area}, \texttt{service lane}, \texttt{tree-lined road} \\
    \textcolor{gray}{\textbf{Prompt:}} \texttt{a driving scene on a \#1}

    \item[\texttt{traffic\_density}:] \texttt{sparse}, \texttt{light}, \texttt{moderate}, \texttt{heavy} \\
    \textcolor{gray}{\textbf{Prompt:}} \texttt{a driving scene with \#1 traffic}

    \item[\texttt{lighting\_condition}:] \texttt{bright}, \texttt{normal}, \texttt{dim}, \texttt{dark} \\
    \textcolor{gray}{\textbf{Prompt:}} \texttt{a driving scene under \#1 lighting condition}

    \item[\texttt{weather}:] \texttt{clear}, \texttt{cloudy}, \texttt{rainy}, \texttt{snowy}, \texttt{foggy} \\
    \textcolor{gray}{\textbf{Prompt:}} \texttt{a driving scene with \#1 weather}
\end{description}
\end{tcolorbox}
\end{table}"""

content = content.replace(table_block, replacement)

with open("main.tex", "w") as f:
    f.write(content)

