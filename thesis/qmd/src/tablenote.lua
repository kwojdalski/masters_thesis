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

  Left-aligned, not centered: WNE UW punkt 10/11 only require the TABLE
  or FIGURE object itself to be centered ("Tabela wyśrodkowana...",
  "Rysunki środkować...") -- the title/caption is explicitly required to
  start at the left margin ("Tytuł od lewego marginesu" / "Podpis... od
  lewego marginesu"), and the source/legend/note text is introduced in
  the same breath as the title with no separate alignment rule of its
  own, so it follows the title's left-margin convention, not the
  object's centering.

  Suppressing the indent: pandoc always serializes the RawBlock and the
  div's actual Para content as separate blocks, joined by a blank line
  -- so a raw `\noindent` placed in the leading RawBlock is orphaned by
  that blank line (\noindent only cancels the indent of the paragraph
  that begins immediately after it; a blank line already starts a new
  \par before the div's own content is reached, and \noindent has
  nothing left to act on). `\parindent0pt`, scoped to the whole group,
  is not paragraph-position-sensitive the same way and reliably
  suppresses indentation for every paragraph typeset inside the group.

  Overriding an ancestor \centering: pandoc renders a table two different
  ways depending on its size. A wide/many-row table becomes `longtable`,
  which is not a floating environment -- nothing after \end{longtable}
  is nested inside anything, so this Div's content lands as an ordinary
  top-level paragraph. A short/narrow table instead becomes a `table`
  float with `\centering{...}` wrapping its body, and because pandoc
  doesn't close that float before inserting the next block, this Div's
  content ends up nested INSIDE that same \centering{} group -- so it
  renders centered despite \parindent0pt (which only ever controls
  indentation, not justification). `\raggedright` here overrides any
  such ancestor \centering for this group's own paragraphs, and is a
  no-op in the ordinary longtable case where no ancestor centering
  exists to override.
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
    -- Annex D italicises the label alone: its source lines measure
    -- "Source:" at I 10.0pt followed by the reference at R 10.0pt, and
    -- nothing on them is bold ("The source should be written in italics
    -- \"Source:\", using the 10 point font", p. 5). Wrapping the whole block
    -- in \itshape slanted the reference text too, and the Markdown
    -- "**Source:**" added bold on top, giving a bold-italic label. So: no
    -- blanket \itshape, and the bold label becomes italic instead.
    blocks:insert(pandoc.RawBlock("latex", "\\nopagebreak\\vspace{-0.3em}{\\footnotesize\\parindent0pt\\raggedright\\relax"))
    blocks:extend(el.content:walk{
      Strong = function(s) return pandoc.Emph(s.content) end,
    })
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
