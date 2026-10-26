"""
PageRank with YOUR OWN web graph.
Run:  python pagerank_custom.py
You choose: how many pages, which page links to which, and the damping factor.
Requires: numpy (matplotlib optional, for plots)
"""
import string
import numpy as np

np.set_printoptions(precision=4, suppress=True)


def header(title):
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


# ----------------------------------------------------------------------
# Input helpers
# ----------------------------------------------------------------------
def ask_int(prompt, lo, hi):
    while True:
        try:
            v = int(input(prompt).strip())
            if lo <= v <= hi:
                return v
        except ValueError:
            pass
        print(f"  Please enter a whole number between {lo} and {hi}.")


def ask_float(prompt, default, lo, hi):
    while True:
        s = input(prompt).strip()
        if s == "":
            return default
        try:
            v = float(s)
            if lo <= v <= hi:
                return v
        except ValueError:
            pass
        print(f"  Please enter a number between {lo} and {hi} (or press Enter for {default}).")


def get_graph():
    n = ask_int("How many pages? (2-26): ", 2, 26)
    pages = list(string.ascii_uppercase[:n])
    print(f"\nPages are named: {', '.join(pages)}")
    print("For each page, type the pages it links TO, separated by commas or spaces.")
    print("Leave blank for no outgoing links. A page may link to itself.\n")

    links = {}
    for p in pages:
        while True:
            raw = input(f"  {p} links to: ").replace(",", " ").upper().split()
            bad = [t for t in raw if t not in pages]
            if bad:
                print(f"    Unknown page(s) {bad}. Choose from {pages}.")
                continue
            links[p] = list(dict.fromkeys(raw))  # remove duplicates, keep order
            break
    return pages, links


# ----------------------------------------------------------------------
# Core PageRank maths
# ----------------------------------------------------------------------
def build_matrix(pages, links):
    """M[i, j] = 1/outdegree(j) if page j links to page i (column-stochastic)."""
    idx = {p: k for k, p in enumerate(pages)}
    M = np.zeros((len(pages), len(pages)))
    for src, dsts in links.items():
        for dst in dsts:
            M[idx[dst], idx[src]] = 1 / len(dsts)
    return M


def google_matrix(M, d):
    M = M.copy()
    n = M.shape[0]
    dangling = M.sum(axis=0) == 0
    M[:, dangling] = 1 / n  # dead ends "link" to every page equally
    return d * M + (1 - d) / n * np.ones((n, n))


def power_iteration(A, tol=1e-12, max_iter=10000):
    n = A.shape[0]
    r = np.ones(n) / n
    hist = [r.copy()]
    for it in range(1, max_iter + 1):
        r_new = A @ r
        hist.append(r_new.copy())
        if np.abs(r_new - r).sum() < tol:
            return r_new, it, np.array(hist)
        r = r_new
    return r, max_iter, np.array(hist)


def strongly_connected(M):
    """True if every page can reach every other page by following links."""
    n = M.shape[0]
    reach = (M > 0).astype(int) + np.eye(n, dtype=int)
    for _ in range(n):
        reach = ((reach @ reach) > 0).astype(int)
    return bool(reach.all())


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    header("Build your web")
    pages, links = get_graph()
    d = ask_float("\nDamping factor d (press Enter for 0.85): ", 0.85, 0.0, 0.999)
    n = len(pages)

    # 1. Graph + matrix ------------------------------------------------
    header("1. Your graph")
    for p in pages:
        print(f"  {p} -> {links[p] if links[p] else '(no links: dead end)'}")

    M = build_matrix(pages, links)
    header("2. Transition matrix M  (column j = where page j sends its vote)")
    print("      " + "     ".join(pages))
    for i, p in enumerate(pages):
        print(f"  {p}  " + "  ".join(f"{v:5.3f}" for v in M[i]))
    dead = [p for p in pages if not links[p]]
    print("\nColumn sums:", M.sum(axis=0))

    # 2. Diagnose problems --------------------------------------------
    header("3. Diagnosis")
    if dead:
        print(f"  Dead end(s): {dead} -> their columns are all 0, so rank leaks out of M.")
    if not strongly_connected(M):
        print("  Graph is NOT strongly connected -> there may be spider traps or")
        print("  unreachable pages, so plain M has no unique PageRank vector.")
    if not dead and strongly_connected(M):
        print("  Graph is strongly connected with no dead ends. M alone is usable,")
        print("  but damping still guarantees convergence.")
    print(f"  Fix: G = d*M + (1-d)/n * ones with d = {d}  "
          "(dead ends link to everyone, plus teleportation).")

    G = google_matrix(M, d)
    assert np.allclose(G.sum(axis=0), 1)

    # 3. Eigen-analysis ------------------------------------------------
    header("4. Eigenvalues of the Google matrix G")
    vals, vecs = np.linalg.eig(G)
    order = np.argsort(-np.abs(vals))
    print("  |eigenvalues| (largest first):", np.abs(vals[order]).round(4))
    print(f"  Dominant eigenvalue = {vals[order[0]].real:.6f}  (should be 1)")
    if n > 1:
        print(f"  Second largest |lambda| = {np.abs(vals[order[1]]):.4f}  (<= d = {d})")

    # 4. Three ways to solve ------------------------------------------
    header("5. Computing PageRank three ways")
    r_power, iters, hist = power_iteration(G)

    A = G - np.eye(n)
    A[-1, :] = 1                      # replace one equation with sum(x) = 1
    b = np.zeros(n)
    b[-1] = 1
    r_direct = np.linalg.solve(A, b)

    _, _, Vt = np.linalg.svd(G - np.eye(n))
    r_null = Vt[-1] / Vt[-1].sum()

    print(f"  Power iteration ({iters} its): {r_power.round(5)}")
    print(f"  Direct solve (G-I)x=0:        {r_direct.round(5)}")
    print(f"  Null space (SVD):             {r_null.round(5)}")
    print("  All agree:", np.allclose(r_power, r_direct, atol=1e-8)
          and np.allclose(r_direct, r_null, atol=1e-8))

    # 5. Results -------------------------------------------------------
    header("6. Final PageRank ranking")
    in_deg = {p: sum(p in links[q] for q in pages) for p in pages}
    print(f"  {'Rank':<6}{'Page':<6}{'Score':<10}{'Inbound links'}")
    for pos, k in enumerate(np.argsort(-r_power), 1):
        print(f"  {pos:<6}{pages[k]:<6}{r_power[k]:<10.4f}{in_deg[pages[k]]}")
    print(f"\n  Sum of ranks = {r_power.sum():.6f}")
    print(f"  Minimum possible rank (teleportation floor) = {(1 - d) / n:.4f}")

    # 6. Damping experiment -------------------------------------------
    header("7. Damping factor vs iterations (your graph)")
    for dd in [0.5, 0.7, 0.85, 0.95, 0.99]:
        _, it, _ = power_iteration(google_matrix(M, dd))
        print(f"  d = {dd:<5} -> {it} iterations")

    # 7. Plots ---------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 3, figsize=(15, 4))
        for i, p in enumerate(pages):
            ax[0].plot(hist[:, i], label=f"Page {p}")
        ax[0].set(title="Power iteration: ranks converge",
                  xlabel="iteration", ylabel="rank")
        if n <= 10:
            ax[0].legend()

        ax[1].bar(pages, r_power, color="#3b78c4")
        ax[1].axhline((1 - d) / n, color="red", ls="--", label="teleport floor")
        ax[1].set(title="Final PageRank", ylabel="rank")
        ax[1].legend()

        ds = np.linspace(0.1, 0.99, 30)
        its = [power_iteration(google_matrix(M, x))[1] for x in ds]
        ax[2].plot(ds, its, "o-")
        ax[2].set(title="Damping factor vs convergence", xlabel="d",
                  ylabel="iterations")

        plt.tight_layout()
        plt.savefig("pagerank_custom_plot.png", dpi=150)
        print("\nSaved plots to pagerank_custom_plot.png")
    except ImportError:
        print("\n(matplotlib not installed; skipping plots)")


if __name__ == "__main__":
    main()
