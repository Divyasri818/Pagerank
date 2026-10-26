# Pagerank
An interactive Streamlit app that shows how Google's PageRank works.

## What it does
- Draw your own web of up to 12 pages by ticking links in a table
- Change the damping factor and see the ranks update live
- Compares power iteration with a direct solve of (G − I)x = 0
- Shows how many iterations it takes to converge
- Shows dominant and second eigenvalues
- Warns about dead ends and graphs that are not strongly connected
- Draws the link graph with circles sized by rank
- Plots damping factor against iterations
- Includes presets: Example, Dead end, Spider trap, Everyone links to A

## Files
- `app.py`: the Streamlit interface
- `pagerank_custom.py`: the PageRank maths (matrix building, Google matrix, power iteration, connectivity check)

## Run
```bash
pip install streamlit numpy pandas graphviz
streamlit run app.py
```

<img width="1917" height="867" alt="image" src="https://github.com/user-attachments/assets/eec45bbf-4a4b-415e-8647-6487a58172cc" />

<img width="1917" height="880" alt="image" src="https://github.com/user-attachments/assets/42b45954-8f9c-4013-b142-ce77f7c166a5" />
