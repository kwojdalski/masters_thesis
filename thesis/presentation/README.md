# Five-minute thesis defence

Seven main slides, followed by three appendix slides for questions. The design
uses the official WNE English identity, white backgrounds, burgundy headings,
and a faculty footer, following the supplied lecture-slide reference.

## Build and present

From the repository root:

```sh
uv sync --extra dev
uv run quarto render thesis/presentation/defence.qmd --to revealjs
uv run quarto preview thesis/presentation/defence.qmd --port 4200 --no-browser
```

Open the preview URL in a browser. Press **F** for full screen, **S** for speaker
view, and the arrow keys to advance. Speaker view contains a complete spoken
script and a time budget for each main slide. Present through slide **07 / 07**;
use the links there to reach the appendix during questions.

`defence.html` embeds the theme, figures, logo and Reveal.js, so the audience deck
can also be opened directly without network access. Use the local preview server
for speaker view: browser restrictions can prevent notes popups from working
when the HTML is opened as a local file. The complete script is also readable in
the `.notes` blocks in `defence.qmd`.

For a PDF, with Google Chrome installed, run:

```sh
uv run --with playwright==1.63.0 python thesis/presentation/export_pdf.py
```

This verifies ten slides, ten notes, and seven timings totalling 300 seconds,
then exports `defence.pdf` without network access. It also runs in CI.

Alternatively, open the rendered presentation with `?print-pdf` appended to the URL
and print to PDF in Chrome/Chromium (landscape, background graphics enabled,
no browser headers or footers). The output contains ten pages: seven main slides
and three appendix pages. Print pages 1–7 for the timed talk alone.

These controls and export settings follow the [Quarto presentation guide](https://quarto.org/docs/presentations/revealjs/presenting.html).

## Rehearsal

| Slide | Focus | Time | Finish at |
|---|---|---:|---:|
| 1 | Question and headline finding | 0:20 | 0:20 |
| 2 | Agent, state, reward and assumptions | 0:45 | 1:05 |
| 3 | Data and chronological split | 0:40 | 1:45 |
| 4 | Learned policies versus random | 0:50 | 2:35 |
| 5 | Estimated spread costs | 1:05 | 3:40 |
| 6 | H2–H4 verdicts | 0:40 | 4:20 |
| 7 | Contribution, limitations and next test | 0:40 | 5:00 |

The script is about 680 spoken words. Rehearse at a comfortable pace, leaving
short pauses on the two charts. Timings are a rehearsal target; slides advance
manually. If running late, shorten the method explanation and preserve the
spread-cost result and conclusion.

## Evidence and reproducibility

Rendering executes five small Python cells using the existing project dependencies.
No training, data download, or live experiment store is needed. `figures.py`
reads the same exported metrics used by the thesis, checks the H1 split and
matching per-symbol evaluation lengths, and plots with plotnine. It does not
recompute trading statistics.

All input paths below are relative to `thesis/qmd/results/`:

| Display | Committed source |
|---|---|
| Data counts | `pooled_td3_hft_lob_state_space_pooled_streaming_selected_dsr/peek/splits.json` |
| Profit-factor chart and H1 table | `pooled_{td3,ddpg,ppo,random}_hft_lob_state_space_pooled_streaming_selected_dsr/latest_finished/evaluation_report.json` |
| Spread chart and cost table | `pooled_td3_hft_lob_state_space_pooled_streaming_selected_dsr/peek/bid_ask_repricing.json` |
| H2–H4 interpretation | Thesis §§6.2–6.4 and §7.1 |
| Method and limitations | Thesis Chapters 3–5 and 7 |

H1 figures are means of separate per-symbol metrics, not a portfolio backtest.
Spread repricing uses total absolute position changes and a mean half-spread;
it is an indicative fixed-path estimate, not fill-by-fill execution or retraining.
The single-seed feature and reward failures leave those comparisons unresolved.

The GitHub Pages workflow renders and publishes the
[HTML deck](https://kwojdalski.github.io/masters_thesis/defence.html) and
[PDF](https://kwojdalski.github.io/masters_thesis/defence.pdf) after a successful
build on `master`. Generated HTML, PDF and figure sidecars remain ignored.

Logo provenance and the faculty's usage conditions are recorded in
[assets/README.md](assets/README.md). For later substantive prose revisions, the
project's thesis-writing skill recommends a Hemingway pass; there are no equation
derivations in the five-minute deck.
