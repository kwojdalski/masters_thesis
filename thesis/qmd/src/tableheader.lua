--[[
tableheader.lua — Pandoc Lua filter enforcing bold, centered table headers.

WNE UW formal requirement (rule 10): "The table's header row is bold and
centered within the cells." Applies to every native Pandoc Table element
(markdown pipe tables, and tables produced from an embedded HTML <table>
via thesis_tables.display_df()). Hand-authored raw-LaTeX tables in
thesis_tables.py bypass the Pandoc AST entirely and are styled directly
in Python instead — this filter cannot see those.

PDF/LaTeX output only; HTML tables keep their existing header styling.

Centering mechanism — pandoc's LaTeX writer renders a table in one of two
incompatible ways depending on each column's colspec width, so a single
centering technique cannot cover both (verified empirically against
pandoc 3.10 by inspecting generated .tex for both cases):

1. Explicit-width columns (colspec width ~= ColWidthDefault, e.g. a wide
   pipe table pandoc renders with p{width} columns): every cell is wrapped
   in its own `\begin{minipage}...\end{minipage}`, and the alignment
   command written inside that minipage (\raggedright / \centering) is
   taken from the Cell's own `alignment` field. Setting
   `cell.alignment = pandoc.AlignCenter` is correct and sufficient here.

2. Default-width columns (colspec width == ColWidthDefault, e.g. a plain
   "llll" table): column alignment is baked into the tabular preamble
   once for the whole column, and a Cell's own `alignment` field is
   silently ignored for ordinary (non-spanning) cells — setting it has
   zero effect on the rendered PDF. The only way to center one cell
   independent of its column's declared alignment is to wrap its content
   in `\multicolumn{1}{c}{...}`, emitted here as raw LaTeX. (Using this
   same multicolumn wrapping on case 1 breaks the build: pandoc always
   wraps p{width}-column cells in a minipage regardless of their content,
   and \multicolumn is invalid nested inside a minipage — "Misplaced
   \omit".)
--]]

local function header_cell_inlines(cell)
  local inlines = pandoc.List({})
  for _, block in ipairs(cell.contents) do
    if block.t == "Plain" or block.t == "Para" then
      for _, inline in ipairs(block.content) do
        inlines:insert(inline)
      end
    end
  end
  return inlines
end

local function is_default_width(colspec)
  local width = colspec[2]
  return width == nil or width == pandoc.ColWidthDefault
end

function Table(el)
  if not (quarto.doc.is_format("pdf") or quarto.doc.is_format("latex")) then
    return nil
  end
  if not el.head or #el.head.rows == 0 then
    return nil
  end

  for _, row in ipairs(el.head.rows) do
    for col_idx, cell in ipairs(row.cells) do
      local inlines = header_cell_inlines(cell)
      if #inlines > 0 then
        local colspec = el.colspecs[col_idx]
        if colspec and is_default_width(colspec) then
          -- Case 2: bake centering into the cell's own LaTeX content.
          cell.contents = pandoc.List({
            pandoc.Plain({
              pandoc.RawInline("latex", "\\multicolumn{1}{c}{"),
              pandoc.Strong(inlines),
              pandoc.RawInline("latex", "}"),
            }),
          })
        else
          -- Case 1: the writer's per-cell minipage already honors this.
          cell.alignment = pandoc.AlignCenter
          cell.contents = pandoc.List({ pandoc.Plain(pandoc.Strong(inlines)) })
        end
      end
    end
  end

  return el
end
