import re

with open("main.tex", "r") as f:
    content = f.read()

# 1. Proposition 1 in Appendix
prop1_app = r"""\textbf{Proposition 1 (Attribute-agnostic AL requires target alignment).} \textit{Consider any AL selector that, within class $c$, chooses $B_c$ source examples using only non-attribute information $h_i$, such as uncertainty, logits, embeddings, losses, detector difficulty, coverage, or image distances, and does not use the attribute values $\{v_{i,a}:a\in\attrs\}$. Let $\dataset_{\mathrm{AL}}$ denote its selected source subset. If the target-slice indicator $\mathbf{1}[x_i\in T_c]$ is conditionally independent of $h_i$ given class $c$, with ties and internal randomness independent of the target indicator, then
\begin{equation*}
\mathbb{E}\big[|\dataset_{\mathrm{AL}}\cap T_c|\big]
=
B_c\frac{|T_c|}{N_c}.
\end{equation*}
Thus an attribute-agnostic AL method can enrich the target slice only when its non-attribute signal is additionally aligned with that slice.}"""

prop1_app_box = r"""\begin{tcolorbox}[
    colback=blue!2!white,
    colframe=blue!20!white,
    arc=3pt,
    boxrule=1pt,
    drop shadow,
    breakable
]
""" + prop1_app + "\n\\end{tcolorbox}"

content = content.replace(prop1_app, prop1_app_box)

# 2. Proof of Proposition 1
proof1 = r"""\begin{proof}[Proof of Proposition~\ref{prop:al-target-alignment}]
Conditional independence makes the target indicators exchangeable with respect to any ranking, batch construction, or randomized choice computed from $h_i$ inside class $c$. The $B_c$ selected positions therefore contain, in expectation, the same target-slice fraction as the class pool itself, namely $|T_c|/N_c$. This argument does not depend on the particular AL heuristic. Uncertainty sampling, margin sampling, loss-based selection, diversity sampling, CoreSet, BADGE, PPAL-style uncertainty-then-diversity selection, and Uncertainty Herding-style uncertainty coverage all fall under the claim whenever their selection variables omit the current attribute metadata and are not otherwise aligned with the target slice.

As a concrete coverage-based instance, let $z_i$ be the representation used by a diversity or coverage selector, such as cluster-based Diversity, CoreSet in feature space, or BADGE in gradient-embedding space. CoreSet-style $k$-center selection minimizes
\begin{equation}
\label{eq:k-center-objective}
\max_i \min_{j\in S} d(z_i,z_j),
\end{equation}
so achieving objective value at most $\epsilon$ is equivalent to constructing an $\epsilon$-cover of the source representations. For a finite set $A$ in this representation space, define the empirical covering number
\begin{equation}
\label{eq:empirical-covering-number}
\mathcal{N}^{\mathrm{emp}}_{\epsilon}(A)
=
\min\big\{|C|: C\subseteq A,\; A\subseteq \cup_{z\in C}B(z,\epsilon)\big\}.
\end{equation}
If the target slice in class $c$ has empirical covering number
\begin{equation}
\label{eq:target-covering-number}
q_c=\mathcal{N}^{\mathrm{emp}}_{\epsilon}\big(\{z_i:x_i\in T_c\}\big),
\end{equation}
then an $\epsilon$-cover objective can cover the entire target region using only $q_c$ selected target representatives. Consequently, when the non-target region still contains uncovered $\epsilon$-scale cells, a coverage-based selector has an optimal solution with at most $q_c$ selected target examples, even if $|T_c|\gg q_c$.

For this coverage-based instance, take $h_i=z_i$. By definition of the empirical covering number, there exists a subset $C_T\subseteq \{z_i:x_i\in T_c\}$ with $|C_T|=q_c$ such that every target-slice representation is within distance $\epsilon$ of some point in $C_T$. Selecting the corresponding $q_c$ target examples therefore makes the whole target slice $\epsilon$-covered. Any additional target example only refines coverage inside an already covered target region. If non-target $\epsilon$-scale cells remain uncovered, a coverage or diversity objective can spend the remaining representatives outside $T_c$ without increasing its maximum covering radius. Thus feature-space coverage does not force more than $q_c$ target selections.

The same conclusion follows from the usual volumetric intuition. If a class region has effective dimension $d$, volume $V$, and the target subregion has volume $V_T$, then away from boundary effects an $\epsilon$-cover requires on the order of $V/(v_d\epsilon^d)$ cells for the full class region but only $V_T/(v_d\epsilon^d)$ cells for the target region, where $v_d$ is the volume of the unit $d$-ball. When the bug slice is semantically important but geometrically compact, $V_T\ll V$, so the number of representatives allocated by a geometry-only objective can be small. The objective allocates capacity to representation volume, not to semantic repair importance.
\end{proof}"""

proof1_box = r"""\begin{tcolorbox}[
    colback=green!2!white,
    colframe=green!20!white,
    arc=3pt,
    boxrule=1pt,
    drop shadow,
    breakable
]
""" + proof1 + "\n\\end{tcolorbox}"

content = content.replace(proof1, proof1_box)

# 3. Proposition 2
prop2 = r"""\begin{proposition}[Rarity-dominates-penalty margin]
\label{prop:rarity-margin}
Assume $\alpha>0$ and $A_{ij}>0$ for every target/non-target pair $(x_i,x_j)$ within class $c$. Define
\begin{equation}
\label{eq:lambda-star}
\lambda_c^\star
=
\min_{x_i\in T_c,\;x_j\notin T_c,\;F_{ij}>0}
\frac{A_{ij}}{F_{ij}},
\end{equation}
with $\lambda_c^\star=\infty$ if no pair has $F_{ij}>0$. For any $0\leq\lambda<\lambda_c^\star$, every target example in $T_c$ has larger base score $\bar{s}_i$ than every non-target example from the same class. In particular, when each target attribute value is rarer than the competing non-target value, i.e., $n_a(v_{i,a}\mid c)<n_a(v_{j,a}\mid c)$ for all $a\in\mathcal{D}_{ij}$, the condition $A_{ij}>0$ holds automatically.
\end{proposition}"""

prop2_box = r"""\begin{tcolorbox}[
    colback=blue!2!white,
    colframe=blue!20!white,
    arc=3pt,
    boxrule=1pt,
    drop shadow,
    breakable
]
""" + prop2 + "\n\\end{tcolorbox}"

content = content.replace(prop2, prop2_box)

# 4. Proof of Proposition 2
proof2 = r"""\begin{proof}
For a target/non-target pair $(x_i,x_j)$ in class $c$, attributes outside $\mathcal{D}_{ij}$ contribute the same score to both examples and cancel. The remaining base-score gap is
\begin{equation}
\label{eq:base-score-gap}
\bar{s}_i-\bar{s}_j
=
A_{ij}-\lambda F_{ij}.
\end{equation}
If $F_{ij}\leq 0$, then $\bar{s}_i-\bar{s}_j\geq A_{ij}>0$ for any $\lambda\geq 0$, so the frequency penalty does not overturn the rarity advantage. If $F_{ij}>0$, then $\bar{s}_i-\bar{s}_j>0$ whenever $\lambda<A_{ij}/F_{ij}$. Taking the minimum over all target/non-target pairs with $F_{ij}>0$ gives the interval $0\leq\lambda<\lambda_c^\star$, under which every pair has a strictly positive score gap. Finally, if $n_a(v_{i,a}\mid c)<n_a(v_{j,a}\mid c)$ and $\alpha>0$, then each rarity term in $A_{ij}$ is positive, so their sum is positive.
\end{proof}"""

proof2_box = r"""\begin{tcolorbox}[
    colback=green!2!white,
    colframe=green!20!white,
    arc=3pt,
    boxrule=1pt,
    drop shadow,
    breakable
]
""" + proof2 + "\n\\end{tcolorbox}"

content = content.replace(proof2, proof2_box)

# 5. Change KITTI from table to figure
# Change the reference Table~\ref{tab:kitti-attribute-corpus} to Figure~\ref{tab:kitti-attribute-corpus}
content = content.replace(r"Table~\ref{tab:kitti-attribute-corpus}", r"Figure~\ref{tab:kitti-attribute-corpus}")

# Change \begin{table}[ht] and \end{table} for tab:kitti-attribute-corpus
kitti_block_old = r"""\begin{table}[ht]
\centering
\caption{KITTI object-detection attribute corpus used for VLM assignment.}
\label{tab:kitti-attribute-corpus}
\begin{tcolorbox}["""

kitti_block_new = r"""\begin{figure}[htbp]
\centering
\caption{KITTI object-detection attribute corpus used for VLM assignment.}
\label{tab:kitti-attribute-corpus}
\begin{tcolorbox}["""

content = content.replace(kitti_block_old, kitti_block_new)

# The end of the block has \end{description}\n\end{tcolorbox}\n\end{table}
# We only want to replace the \end{table} that matches this block. Let's do it with a more specific replace.
kitti_end_old = r"""    \textcolor{gray}{\textbf{Prompt:}} \texttt{a driving scene in \#1 weather}
\end{description}
\end{tcolorbox}
\end{table}"""

kitti_end_new = r"""    \textcolor{gray}{\textbf{Prompt:}} \texttt{a driving scene in \#1 weather}
\end{description}
\end{tcolorbox}
\end{figure}"""

content = content.replace(kitti_end_old, kitti_end_new)

with open("main.tex", "w") as f:
    f.write(content)

