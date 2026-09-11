import sys

with open('main.tex', 'r') as f:
    content = f.read()

# 1. Revert VLM box
vlm_old = r"""\begin{tcolorbox}[
    colback=blue!3!white,
    colframe=blue!30!black,
    title=VLM Role and Leakage Control,
    fonttitle=\bfseries,
    rounded corners,
    drop shadow,
    breakable
]
\textbf{Where VLM annotations enter.}"""
vlm_new = r"""\paragraph{VLM Role and Leakage Control}
\textbf{Where VLM annotations enter.}"""
content = content.replace(vlm_old, vlm_new)

# Remove the \end{tcolorbox} for VLM box
vlm_end_old = r"""independent human labels.
\end{tcolorbox}

\begin{tcolorbox}["""
vlm_end_new = r"""independent human labels.

\begin{tcolorbox}["""
content = content.replace(vlm_end_old, vlm_end_new)

# 2. Revert Human Audit Protocol box
audit_old = r"""\begin{tcolorbox}[
    colback=green!3!white,
    colframe=green!40!black,
    title=Human Audit Protocol,
    fonttitle=\bfseries,
    rounded corners,
    drop shadow,
    breakable
]
We conduct"""
audit_new = r"""\paragraph{Human Audit Protocol}
We conduct"""
content = content.replace(audit_old, audit_new)

audit_end_old = r"""these attributes to identify repair-relevant source examples.
\end{tcolorbox}

\begin{table}[h]"""
audit_end_new = r"""these attributes to identify repair-relevant source examples.

\begin{table}[h]"""
content = content.replace(audit_end_old, audit_end_new)

# 3. Add Proposition 1 and its proof formatting in appendix
prop1_proof_old = r"""\paragraph{Proof of Proposition~\ref{prop:al-target-alignment}}
Conditional independence"""
prop1_proof_new = r"""\begin{tcolorbox}[
    colback=blue!2!white,
    colframe=blue!20!white,
    arc=3pt,
    boxrule=1pt,
    drop shadow,
    breakable
]
\textbf{Proposition 1 (Attribute-agnostic AL requires target alignment).} \textit{Consider any AL selector that, within class $c$, chooses $B_c$ source examples using only non-attribute information $h_i$, such as uncertainty, logits, embeddings, losses, detector difficulty, coverage, or image distances, and does not use the attribute values $\{v_{i,a}:a\in\attrs\}$. Let $\dataset_{\mathrm{AL}}$ denote its selected source subset. If the target-slice indicator $\mathbf{1}[x_i\in T_c]$ is conditionally independent of $h_i$ given class $c$, with ties and internal randomness independent of the target indicator, then
\begin{equation*}
\mathbb{E}\big[|\dataset_{\mathrm{AL}}\cap T_c|\big]
=
B_c\frac{|T_c|}{N_c}.
\end{equation*}
Thus an attribute-agnostic AL method can enrich the target slice only when its non-attribute signal is additionally aligned with that slice.}
\end{tcolorbox}

\begin{tcolorbox}[
    colback=green!2!white,
    colframe=green!20!white,
    arc=3pt,
    boxrule=1pt,
    drop shadow,
    breakable
]
\begin{proof}[Proof of Proposition~\ref{prop:al-target-alignment}]
Conditional independence"""
content = content.replace(prop1_proof_old, prop1_proof_new)

prop1_proof_end_old = r"""semantic repair importance.

\paragraph{Rarity and generated-data penalty}"""
prop1_proof_end_new = r"""semantic repair importance.
\end{proof}
\end{tcolorbox}

\paragraph{Rarity and generated-data penalty}"""
content = content.replace(prop1_proof_end_old, prop1_proof_end_new)

with open('main.tex', 'w') as f:
    f.write(content)
print("Updated main.tex")
