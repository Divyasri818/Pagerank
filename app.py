"""
PageRank Explorer - Streamlit app.
Uses the maths from pagerank_custom.py (build_matrix, google_matrix,
power_iteration, strongly_connected), so the UI runs YOUR Python code.

Run:   streamlit run app.py
"""
import numpy as np
import pandas as pd
import streamlit as st

import pagerank_custom as pr  # <- your existing file, same folder

MAX_PAGES = 12
LETTERS = "ABCDEFGHIJKL"

# (from, to) index pairs for each preset
PRESETS = {
    "Example": [(0, 1), (0, 2), (1, 2), (2, 0), (3, 2), (3, 4), (4, 3)],
    "Dead end": [(0, 1), (0, 2), (1, 2), (3, 0), (3, 2), (4, 0)],
    "Spider trap": [(0, 1), (0, 2), (1, 2), (2, 2), (3, 0), (4, 3)],
    "Everyone links to A": [(1, 0), (2, 0), (3, 0), (4, 0), (0, 1)],
}

st.set_page_config(page_title="PageRank Explorer", page_icon="🕸️", layout="wide")


# ----------------------------------------------------------------------
# State: L[i, j] = True if page i links to page j
# ----------------------------------------------------------------------
def load_preset(name):
    L = np.zeros((MAX_PAGES, MAX_PAGES), dtype=bool)
    for i, j in PRESETS[name]:
        L[i, j] = True
    st.session_state.L = L
    st.session_state.n = 5
    st.session_state.version = st.session_state.get("version", 0) + 1


if "L" not in st.session_state:
    load_preset("Example")

# ----------------------------------------------------------------------
# Sidebar controls
# ----------------------------------------------------------------------
st.sidebar.header("Build your web")
n = st.sidebar.slider("Number of pages", 2, MAX_PAGES, key="n")
d = st.sidebar.slider("Damping factor d", 0.0, 0.99, 0.85, 0.01)

st.sidebar.selectbox("Preset graph", list(PRESETS), key="preset")
# on_click runs BEFORE the widgets above are redrawn, so it may change n safely
st.sidebar.button("Load preset", on_click=lambda: load_preset(st.session_state.preset))

pages = list(LETTERS[:n])

# ----------------------------------------------------------------------
# Title + link editor
# ----------------------------------------------------------------------
st.title("PageRank Explorer")
st.caption("Tick a box to make the row page link to the column page. "
           "Everything below is calculated by pagerank_custom.py.")

df = pd.DataFrame(st.session_state.L[:n, :n], index=pages, columns=pages)
edited = st.data_editor(
    df,
    key=f"editor_{st.session_state.version}_{n}",
    use_container_width=True,
)
st.session_state.L[:n, :n] = edited.to_numpy(dtype=bool)
L = st.session_state.L[:n, :n]

# ----------------------------------------------------------------------
# Maths (reusing pagerank_custom.py)
# ----------------------------------------------------------------------
links = {p: [pages[j] for j in range(n) if L[i, j]] for i, p in enumerate(pages)}
M = pr.build_matrix(pages, links)
G = pr.google_matrix(M, d)
r, iters, hist = pr.power_iteration(G)

A = G - np.eye(n)                 # direct solve of (G - I)x = 0, sum(x) = 1
A[-1, :] = 1
b = np.zeros(n)
b[-1] = 1
r_direct = np.linalg.solve(A, b)
agree = np.allclose(r, r_direct, atol=1e-8)

dead = [p for p in pages if not links[p]]
strong = pr.strongly_connected(M)
eig = np.sort(np.abs(np.linalg.eigvals(G)))[::-1]

# ----------------------------------------------------------------------
# Results
# ----------------------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Iterations to converge", iters)
c2.metric("Dominant eigenvalue", f"{eig[0]:.4f}")
c3.metric("Second eigenvalue", f"{eig[1]:.4f}", help="Always at most d")
c4.metric("Minimum possible rank", f"{(1 - d) / n:.4f}")

left, right = st.columns([3, 2])

with left:
    st.subheader("Link graph")
    top = r.max()
    dot = ["digraph { rankdir=LR; bgcolor=transparent;",
           'node [shape=circle style=filled fontcolor=white fixedsize=true fontsize=12];']
    for i, p in enumerate(pages):
        size = 0.45 + 1.1 * np.sqrt(r[i] / top)
        color = "#c47d00" if p in dead else "#0f8b8d"
        dot.append(f'{p} [width={size:.2f} label="{p}\\n{r[i]*100:.1f}%" fillcolor="{color}"];')
    for p, targets in links.items():
        for t in targets:
            dot.append(f"{p} -> {t};")
    dot.append("}")
    st.graphviz_chart("\n".join(dot), use_container_width=True)
    st.caption("Bigger circles have a higher PageRank. Amber circles are dead ends.")

with right:
    st.subheader("Ranking")
    table = pd.DataFrame({
        "Page": pages,
        "PageRank": r,
        "Inbound links": L.sum(axis=0),
    }).sort_values("PageRank", ascending=False).reset_index(drop=True)
    table.index += 1
    st.dataframe(
        table,
        use_container_width=True,
        column_config={
            "PageRank": st.column_config.ProgressColumn(
                "PageRank", min_value=0.0, max_value=float(top), format="%.4f"),
        },
    )

    st.subheader("What the maths says")
    if dead:
        st.warning(f"Dead end(s): {', '.join(dead)}. With no links out, rank would "
                   "leak away, so they are treated as linking to every page.")
    if not strong:
        st.warning("The graph is not strongly connected, so M alone has no guaranteed "
                   f"unique PageRank. Damping with d = {d:.2f} fixes that.")
    elif not dead:
        st.success("Every page can reach every other page and none is a dead end.")
    if agree:
        st.info("Power iteration and the direct solve of (G − I)x = 0 give the same ranks.")
    else:
        st.error("Power iteration and the direct solve disagree. Try a smaller d.")

st.subheader("Power iteration converging")
st.line_chart(pd.DataFrame(hist, columns=pages))
st.caption("Every page starts equal. Rank flows along the links until the vector "
           "stops changing, which is the eigenvector for λ = 1.")

# ----------------------------------------------------------------------
# Damping experiment + matrices
# ----------------------------------------------------------------------
st.subheader("Damping factor vs iterations")
ds = np.round(np.linspace(0.1, 0.99, 25), 3)
its = [pr.power_iteration(pr.google_matrix(M, x))[1] for x in ds]
st.line_chart(pd.DataFrame({"iterations": its}, index=ds))

with st.expander("Show the matrices"):
    st.write("**Transition matrix M** (column j = where page j sends its vote)")
    st.dataframe(pd.DataFrame(M, index=pages, columns=pages).round(3))
    st.write(f"**Google matrix G = d·M + (1−d)/n** with d = {d:.2f}")
    st.dataframe(pd.DataFrame(G, index=pages, columns=pages).round(4))
