# Fixed two-buffer paper assets

Use this package for the current comparison: every method has two GPU buffers.

- `throughput.pdf` / `.svg` / `.png`: publication figure and preview.
- `throughput.tex` / `.md` / `.csv`: LaTeX table (booktabs), readable table and data.
- `trials.csv`: all 30 selected, anonymized records.
- `paper_results.md`: methods, results, caption and verification limits.
- `build_publication.py`: verifies the included trial CSV hash and regenerates numeric
  assets and the figure using NumPy and Matplotlib.

Regenerate from the project root with:
`python3 docs/benchmarks/two-buffers/build_publication.py`
The anonymized trial CSV is included; the original source ZIP is not required.

The previous all-buffer package remains a historical export. Three-buffer
profiles and replay results are not part of this fixed two-buffer comparison.
