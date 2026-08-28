--[[
tablealign.lua — Pandoc Lua filter right-aligning numeric table columns.

WNE UW rule 10 specifies only three alignments: the table centered between
the side margins, the header row bold and centered in its cells, and the
"Wyszczególnienie" (item/label) column left-aligned where warranted. It says
nothing about numeric values, so right-aligning them is a typographic choice
rather than a formal requirement: digits line up by magnitude and columns of
figures become visually comparable.

Before this filter the thesis was inconsistent about it. Tables emitted by
thesis_tables.comparison_table_html (H2) right-aligned their values, while
those emitted by thesis_tables.display_df (H1, H3, H4, split sizes, raw-file
inventory) inherited pandas' default left alignment — the same chapter
presenting the same kind of figures two different ways.

Alignment mechanism — the LaTeX writer bakes column alignment into the
tabular preamble from each column's colspec, and (for ordinary cells in
default-width columns) silently ignores a Cell's own `alignment` field. See
the extended note in tableheader.lua. Setting the colspec is therefore the
reliable lever, and it works for the HTML writer too, which emits a
corresponding `text-align` style.

Header cells stay centered: tableheader.lua runs before this filter and
pins them independently of the column alignment — via \multicolumn{1}{c}{}
for default-width columns, and the cell's own alignment inside the
per-cell minipage for explicit-width ones.

A column is right-aligned only when every cell in it is either a number or
a neutral placeholder, and at least one is an actual number. Anything else
(a status word, a timestamp, prose) leaves the column untouched.
--]]

-- Values that neither confirm nor disqualify a column as numeric: the
-- missing-data placeholders and empty cells these tables use.
local PLACEHOLDERS = {
  [""] = true, ["—"] = true, ["–"] = true, ["-"] = true,
  ["N/A"] = true, ["n/a"] = true, ["NA"] = true, ["nan"] = true, ["None"] = true,
}

--- Classify one cell's text.
--- @return true numeric, false definitely not numeric, nil neutral/placeholder
local function classify(text)
  local s = text:gsub("%s", "")
  if PLACEHOLDERS[s] then return nil end
  if s == "" then return nil end

  -- Trailing significance markers, e.g. a p-value rendered "0.0234*".
  s = s:gsub("[%*†‡]+$", "")
  s = s:gsub(",", "")            -- thousands separators
  s = s:gsub("%%$", "")          -- trailing percent
  s = s:gsub("^[%-%+]", "")      -- ASCII sign
  s = s:gsub("^\u{2212}", "")    -- U+2212 MINUS SIGN
  s = s:gsub("^\u{2013}", "")    -- U+2013 EN DASH used as a minus

  if s == "" then return nil end
  if s:match("^%d+%.?%d*$") then return true end                    -- 12  12.34
  if s:match("^%.%d+$") then return true end                        -- .34
  if s:match("^%d+%.?%d*[eE][%-%+]?%d+$") then return true end      -- 2.28e-07
  return false
end

local function cell_text(cell)
  return pandoc.utils.stringify(cell.contents)
end

--- Is every body cell in this column numeric-or-placeholder, with >=1 number?
local function column_is_numeric(el, col_idx)
  local saw_number = false
  for _, body in ipairs(el.bodies) do
    for _, row in ipairs(body.body) do
      local cell = row.cells[col_idx]
      if cell then
        local verdict = classify(cell_text(cell))
        if verdict == false then return false end
        if verdict == true then saw_number = true end
      end
    end
  end
  return saw_number
end

function Table(el)
  if not el.colspecs then return nil end

  local changed = false
  for col_idx, colspec in ipairs(el.colspecs) do
    if column_is_numeric(el, col_idx) then
      colspec[1] = pandoc.AlignRight
      changed = true
    end
  end

  if not changed then return nil end
  return el
end
