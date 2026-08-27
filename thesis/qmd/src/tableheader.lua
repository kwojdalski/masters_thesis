--[[
tableheader.lua — Pandoc Lua filter enforcing bold, centered table headers.

WNE UW formal requirement (rule 10): "The table's header row is bold and
centered within the cells." Applies to every native Pandoc Table element
(markdown pipe tables, and tables produced from an embedded HTML <table>
via thesis_tables.display_df()). Hand-authored raw-LaTeX tables in
thesis_tables.py bypass the Pandoc AST entirely and are styled directly
in Python instead — this filter cannot see those.

PDF/LaTeX output only; HTML tables keep their existing header styling.
--]]

local function bold_block(block)
  if block.t == "Plain" or block.t == "Para" then
    return pandoc.Plain(pandoc.Strong(block.content))
  end
  return block
end

function Table(el)
  if not (quarto.doc.is_format("pdf") or quarto.doc.is_format("latex")) then
    return nil
  end
  if not el.head or #el.head.rows == 0 then
    return nil
  end

  for _, row in ipairs(el.head.rows) do
    for _, cell in ipairs(row.cells) do
      cell.alignment = pandoc.AlignCenter
      local new_contents = pandoc.List({})
      for _, block in ipairs(cell.contents) do
        new_contents:insert(bold_block(block))
      end
      cell.contents = new_contents
    end
  end

  return el
end
