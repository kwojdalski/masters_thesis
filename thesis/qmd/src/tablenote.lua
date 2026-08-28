--[[
tablenote.lua — Pandoc Lua filter for standardised table notes.

Usage in Markdown:
  ::: {.table-note}
  **Source:** Author's own synthesis based on [@Author2024].
  **Legend:** yes = included in model; — = not used.
  **Note:** First 500 events skipped for rolling-window warm-up.
  :::

Usage from Python cells (via thesis_tables.table_note()):
  The helper emits display(Markdown(":::{.table-note}\n...\n:::"))
  which Quarto processes through this same filter.

HTML output: wraps the div content in a <div class="table-note"> with
  inline CSS for a consistently small, muted, italic style.
PDF/LaTeX output: wraps the block content in {\footnotesize\itshape …}
  with a small negative vspace to reduce the gap below the table, and
  centers it (via `center`) to match the page's other centered floats
  (longtable defaults to centered; figures are emitted with fig-align:
  center) -- a plain left-flush paragraph here would wrap at the true
  page margin while the table above it sits centered, visually
  misaligned with the table it annotates.
--]]

function Div(el)
  if not el.classes:includes("table-note") then
    return nil  -- leave all other divs unchanged
  end

  -- ── PDF / LaTeX ──────────────────────────────────────────────────────────
  if quarto.doc.is_format("pdf") or quarto.doc.is_format("latex") then
    local blocks = pandoc.Blocks{}
    -- \nopagebreak discourages LaTeX from splitting the page exactly between
    -- the table and its source/legend note; without it the note can be torn
    -- across a page boundary (e.g. a multi-citation Source line split
    -- mid-parenthesis) while the table itself stays fully on the prior page.
    blocks:insert(pandoc.RawBlock("latex", "\\nopagebreak\\vspace{-0.3em}{\\footnotesize\\itshape\\begin{center}"))
    blocks:extend(el.content)
    blocks:insert(pandoc.RawBlock("latex", "\\end{center}}\\vspace{0.3em}"))
    return blocks
  end

  -- ── HTML (default) ───────────────────────────────────────────────────────
  local style = table.concat({
    "font-size:0.875em",
    "color:#484848",
    "font-style:italic",
    "margin-top:0.15em",
    "margin-bottom:0.8em",
    "line-height:1.4",
  }, ";")

  el.attributes["style"] = style
  el.classes = pandoc.List{"table-note"}  -- keep class for any extra CSS targeting
  return el
end
