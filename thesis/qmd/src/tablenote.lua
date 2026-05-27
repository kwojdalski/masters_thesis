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
  with a small negative vspace to reduce the gap below the table.
--]]

function Div(el)
  if not el.classes:includes("table-note") then
    return nil  -- leave all other divs unchanged
  end

  -- ── PDF / LaTeX ──────────────────────────────────────────────────────────
  if quarto.doc.is_format("pdf") or quarto.doc.is_format("latex") then
    local blocks = pandoc.Blocks{}
    blocks:insert(pandoc.RawBlock("latex", "\\vspace{-0.3em}{\\footnotesize\\itshape"))
    blocks:extend(el.content)
    blocks:insert(pandoc.RawBlock("latex", "}\\vspace{0.3em}"))
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
