
# @knitr template_objects
# For package 'rpm'

# data

rmd.template <-
'\n
## Introduction
ADD_TEXT_HERE

### Motivation
ADD_TEXT_HERE

### Details
ADD_TEXT_HERE

#### Capabilities
ADD_TEXT_HERE

#### Limitations
ADD_TEXT_HERE

## Related items

### Files and Data
ADD_TEXT_HERE

### Code flow
ADD_TEXT_HERE

```{r code_sankey, echo=F, eval=T}
```

```{r code_sankey_embed, echo=F, eval=T, comment=NA, results="asis", tidy=F}
```

## R code

### Setup
ADD_TEXT_HERE: EXAMPLE
Setup consists of loading required **R** packages and additional files, preparing any command line arguments for use, and defining functions and other **R** objects.
\n'

# default path
matt.proj.path <- "C:/github"

# @knitr fun_newProject
newProject <- function(name, path,
	dirs=c("code", "data", "docs", "plots", "workspaces"),
	docs.dirs=c("diagrams", "ioslides", "notebook", "Rmd/include", "md", "html", "Rnw", "pdf", "timeline", "tufte"),
	overwrite = FALSE){
	
  stopifnot(is.character(name))
  name <- file.path(path, name)
	if(file.exists(name) && !overwrite) stop("This project already exists.")
  dir.create(name, recursive = TRUE, showWarnings = FALSE)
	if(!file.exists(name)) stop("Directory appears invalid.")
	
  path.dirs <- file.path(name, dirs)
  sapply(path.dirs, dir.create, showWarnings = FALSE)
  path.docs <- file.path(name, "docs", docs.dirs)
	if("docs" %in% dirs) sapply(path.docs, dir.create, recursive=TRUE, showWarnings=FALSE)
	if(overwrite) cat("Project directories updated.\n") else cat("Project directories created.\n")
}

# @knitr fun_rmdHeader
# Generate Rmd files
# Rmd yaml front-matter
# called by genRmd
.rmdHeader <- function(title="filenames", author="Matthew Leonawicz", theme="united", highlight="zenburn", toc=FALSE, keep.md=TRUE, ioslides=FALSE, include.pdf=FALSE){
	if(toc) toc <- "true" else toc <- "false"
	if(keep.md) keep.md <- "true" else keep.md <- "false"
	if(ioslides) hdoc <- "ioslides_presentation" else hdoc <- "html_document"
	rmd.header <- "---\n"
	if(!is.null(title)) rmd.header <- paste0(rmd.header, 'title: ', title, '\n')
	if(!is.null(author)) rmd.header <- paste0(rmd.header, 'author: ', author, '\n')
	rmd.header <- paste0(rmd.header, 'output:\n  ', hdoc, ':\n    toc: ', toc, '\n    theme: ', theme, '\n    highlight: ', highlight, '\n    keep_md: ', keep.md, '\n')
	if(ioslides) rmd.header <- paste0(rmd.header, '    widescreen: true\n')
	if(include.pdf) rmd.header <- paste0(rmd.header, '  pdf_document:\n    toc: ', toc, '\n    highlight: ', highlight, '\n')
	rmd.header <- paste0(rmd.header, '---\n')
	rmd.header
}

# @knitr fun_rmdknitrSetup
# Rmd knitr setup chunk
# called by genRmd
.rmdknitrSetup <- function(file, include.sankey = FALSE){
	x <- paste0('\n```{r knitr_setup, echo=FALSE}\nopts_chunk$set(cache=FALSE, eval=FALSE, tidy=TRUE, message=FALSE, warning=FALSE)\n')
	if(include.sankey) x <- paste0(x, 'read_chunk("../../code/proj_sankey.R")\n')
	x <- paste0(x, 'read_chunk("../../code/', gsub("\\.Rmd", "\\.R", basename(file)), '")\n```\n')
	x
}

# @knitr fun_genRmd
genRmd <- function(
  path, replace = FALSE, 
  header.args=list(title = "filename", author = NULL, theme = "united", highlight = "zenburn", toc = FALSE, keep.md = TRUE, ioslides = FALSE, include.pdf = FALSE),
	update.header=FALSE, ...
){
	
  stopifnot(is.character(path))
  files <- list.files(path, pattern = ".R$", full = TRUE)
  stopifnot(length(files) > 0)
  rmd <- gsub("\\.R", "\\.Rmd", basename(files))
  rmd <- file.path(dirname(path), "docs/Rmd", rmd)
	if(!(replace | update.header)) rmd <- rmd[!sapply(rmd, file.exists)]
	if(update.header) rmd <- rmd[sapply(rmd, file.exists)]
  stopifnot(length(rmd) > 0)
	
	sinkRmd <- function(x, arglist,  ...){
		if(arglist$title=="filename") arglist$title <- gsub("\\.Rmd", "\\.R", basename(x))
		y1 <- do.call(.rmdHeader, arglist)
		y2 <- .rmdknitrSetup(file=x, ...)
		y3 <- list(...)$rmd.template
		if(is.null(y3)) y3 <- rmd.template
		sink(x)
		sapply(c(y1, y2, y3), cat)
		sink()
	}
	
	swapHeader <- function(x, arglist){
		if(arglist$title=="filename") arglist$title <- gsub("\\.Rmd", "\\.R", basename(x))
		header <- do.call(.rmdHeader, arglist)
		l <- readLines(x)
		ind <- which(l=="---")
		l <- l[(ind[2] + 1):length(l)]
		l <- paste0(l, "\n")
		sink(x)
		sapply(c(header, l), cat)
		sink()
	}
	
	if(update.header){
		sapply(rmd, swapHeader, arglist=header.args)
		cat("yaml header updated for each .Rmd file.\n")
	} else {
		sapply(rmd, sinkRmd, arglist=header.args, ...)
		cat(".Rmd files created for each .R file.\n")
	}
}

# @knitr fun_chunkNames
chunkNames <- function(path, rChunkID="# @knitr", rmdChunkID="```{r", append.new=FALSE){
  files <- list.files(path, pattern = ".R$", full = TRUE)
  stopifnot(length(files) > 0)
  l1 <- lapply(files, readLines)
  l1 <- rapply(l1, function(x) x[substr(x, 1, nchar(rChunkID)) == rChunkID], how = "replace")
  l1 <- rapply(l1, function(x, p) gsub(paste0(p, " "), "", x), how = "replace", p = rChunkID)
	if(!append.new) return(l1)
	
  appendRmd <- function(x, rmd.files, rChunks, rmdChunks, ID){
		r1 <- rmdChunks[[x]]
		r2 <- rChunks[[x]]
		r.new <- r2[!(r2 %in% r1)]
		if(length(r.new)){
			r.new <- paste0(ID, " ", r.new, "}\n```\n", collapse="") # Hard coded brace and backticks
			sink(rmd.files[x], append=TRUE)
			cat("\nNEW_CODE_CHUNKS\n")
			cat(r.new)
			sink()
			paste(basename(rmd.files[x]), "appended with new chunk names from .R file")
		}
		else paste("No new chunk names appended to", basename(rmd.files[x]))
	}
	
	rmd <- gsub("\\.R", "\\.Rmd", basename(files))
	rmd <- file.path(dirname(path), "docs/Rmd", rmd)
	rmd <- rmd[sapply(rmd, file.exists)]
	stopifnot(length(rmd) > 0) # Rmd files must exist
	files.ind <- match(gsub("\\.Rmd", "", basename(rmd)), gsub("\\.R", "", basename(files))) # Rmd exist for which R script
	l2 <- lapply(rmd, readLines)
	l2 <- rapply(l2, function(x) x[substr(x, 1, nchar(rmdChunkID))==rmdChunkID], how="replace")
	l2 <- rapply(l2, function(x, p) gsub(paste0(p, " "), "", x), how="replace", p=gsub("\\{", "\\\\{", rmdChunkID))
	l2 <- rapply(l2, function(x) gsub("}", "", sapply(strsplit(x, ","), "[[", 1)), how="replace")
	sapply(1:length(rmd), appendRmd, rmd.files=rmd, rChunks=l1[files.ind], rmdChunks=l2, ID=rmdChunkID)
}

# @knitr fun_swapHeadings
# Rmd <-> Rnw document conversion
# Conversion support functions
# called by .swap()
.swapHeadings <- function(from, to, x){
	nc <- nchar(x)
	ind <- which(substr(x, 1, 1)=="\\")
	if(!length(ind)){ # assume Rmd file
		ind <- which(substr(x, 1, 1)=="#")
		ind.n <- rep(1, length(ind))
		for(i in 2:6){
			ind.tmp <- which(substr(x[ind], 1, i)==substr("######", 1, i))
			if(length(ind.tmp)) ind.n[ind.tmp] <- ind.n[ind.tmp] + 1 else break
		}
		for(i in 1:length(ind)){
			n <- ind.n[i]
			input <- paste0(substr("######", 1, n), " ")
			h <- x[ind[i]]
			h <- gsub("\\*", "_", h) # Switch any markdown boldface asterisks in headings to double underscores
			heading <- gsub("\n", "", substr(h, n+2, nc[ind[i]]))
			#h <- gsub(input, "", h)
			if(n <= 2) subs <- "\\" else if(n==3) subs <- "\\sub" else if(n==4) subs <- "\\subsub" else if(n >=5) subs <- "\\subsubsub"
			output <- paste0("\\", subs, "section{", heading, "}\n")
			x[ind[i]] <- gsub(h, output, h)
		}
	} else { # assume Rnw file
		ind <- which(substr(x, 1, 8)=="\\section")
		if(length(ind)){
			for(i in 1:length(ind)){
				h <- x[ind[i]]
				heading <- paste0("## ", substr(h, 10, nchar(h)-2), "\n")
				x[ind[i]] <- heading
			}
		}
		ind <- which(substr(x, 1, 4)=="\\sub")
		if(length(ind)){
			for(i in 1:length(ind)){
				h <- x[ind[i]]
				z <- substr(h, 2, 10)
				if(z=="subsubsub") {p <- "##### "; n <- 19 } else if(substr(z, 1, 6)=="subsub") { p <- "#### "; n <- 16 } else if(substr(z, 1, 3)=="sub") { p <- "### "; n <- 13 }
				heading <- paste0(p, substr(h, n, nchar(h)-2), "\n")
				x[ind[i]] <- heading
			}
		}
	}
	x
}

# @knitr fun_swapChunks
# Rmd <-> Rnw document conversion
# Conversion support functions
# called by .swap()
# .swapChunks changes code chunks in rnw to code chunks in RMD, e.g. in rnw it should start with <<< and end with @ to be correct
.swapChunks <- function(from, to, x, offset.end=1){
	gsbraces <- function(txt) gsub("\\{", "\\\\{", txt)
	nc <- nchar(x)
	# chunk.start.open <- substr(x, 1, nchar(from[1]))==from[1]

		from %<>% {gsub('(\\{|\\})','\\\\\\1',x=.,perl=T)}
	chunk.start.open <- grepl(paste0('^',gsub('\\s','',from[1])),x =x,perl=T )
	
	
	# chunk.start.close <- substr(x, nc-offset.end-nchar(from[2])+1, nc - offset.end)==from[2]
	chunk.start.close <- grepl(paste0(from[2],'(\n)?$'),x = x,perl=T) 
	chunk.start <- which(chunk.start.open & chunk.start.close)
	chunk.end <- which(substr(x, 1, nchar(from[3]))==from[3])# & nc==nchar(from[3]) + offset.end)
	x[chunk.start] <- gsub(from[2], to[2], gsub(from[1], to[1], x[chunk.start]))
	x[chunk.end] <- gsub(paste0(from[3],'\\n$'), to[3], x[chunk.end])
	if(!length(x[chunk.end])) warning('Chunk end is incorrect')
	if(!length(x[chunk.start])) warning('Chunk start is incorrect')
	chunklines <- as.numeric(unlist(mapply(seq, chunk.start, chunk.end)))
	list(x, chunklines)
}

# @knitr fun_swapEmphasis
# Rmd <-> Rnw document conversion
# Conversion support functions
# called by .swap()
# I know I use '**' strictly for bold font in Rmd files.
# For now, this function assumes:
# 1. The only emphasis in a doc is boldface or typewriter.
# 2. These instances are always preceded by a space, a carriage return, or an open bracket,
# 3. and followed by a space, period, comma, or closing bracket.
.swapEmphasis <- function(x, emphasis="remove",
	pat.remove=c("`", "\\*\\*", "__"),
	pat.replace=pat.remove,
	replacement=c("\\\\texttt\\{", "\\\\textbf\\{", "\\\\textbf\\{", "\\}", "\\}", "\\}")){
	
	stopifnot(emphasis %in% c("remove", "replace"))
	n <- length(pat.replace)
	rep1 <- replacement[1:n]
	rep2 <- replacement[(n+1):(2*n)]
	prefix <- c(" ", "^", "\\{", "\\(")
	suffix <- c(" ", ",", "-", "\n", "\\.", "\\}", "\\)")
	n.p <- length(prefix)
	n.s <- length(suffix)
	pat.replace <- c(paste0(rep(prefix, n), rep(pat.replace, each=n.p)), paste0(rep(pat.replace, each=n.s), rep(suffix, n)))
	replacement <- c(paste0(rep(gsub("\\^", "", prefix), n), rep(rep1, each=n.p)), paste0(rep(rep2, each=n.s), rep(suffix, n)))
	if(emphasis=="remove") for(k in 1:length(pat.remove)) x <- sapply(x, function(v, p, r) gsub(p, r, v), p=pat.remove[k], r="")
	if(emphasis=="replace") for(k in 1:length(pat.replace)) x <- sapply(x, function(v, p, r) gsub(p, r, v), p=pat.replace[k], r=replacement[k])
	x
}

# @knitr fun_swap
# Rmd <-> Rnw document conversion
# Conversion support functions
# called by .convertDocs()

# .swap -------------------------------------------------------------------

.swap <- function(file, header=NULL, outDir, rmdChunkID, rnwChunkID, emphasis, overwrite, ...){
  # Additional arguments
	title <- list(...)$title
	author <- list(...)$author
	highlight <- list(...)$highlight
	standalone <- list(...)$standalone
	is_child <- list(...)$is_child
	ext <- tail(strsplit(file, "\\.")[[1]], 1)
	l <- readLines(file) %>% {.[substr(., 1, 7)!="<style>"]} # Strip any html style lines
	
	
	
	
	# My additional stuff
	if(is.null(is_child)){
	  is_child  <- gsub(pattern = paste0('.*',ext,'\\/'),'',x = file, perl = T) %>%
	  {sapply(regmatches(., gregexpr("\\/", ., perl=T)), length)} %>% as.logical()
	  if(is_child && is.null(standalone)) standalone <- F
	}
	  
	
	# if(!is.null(standalone) && !is.null(is_child) && !is_child) standalone <- FALSE
  if(is.null(standalone)) standalone <- T	
	if(is.null(is_child)) is_child <- F 

	if(ext=="Rmd"){
		from <- rmdChunkID; to <- rnwChunkID
		hl.default <- "solarized-light"
		out.ext <- "Rnw"
		if(!is_child){
  		h.ind <- 1:which(l=="---")[2]
  		h <- l[h.ind]
  		t.ind <- which(substr(h, 1, 7)=="title: ")
  		a.ind <- which(substr(h, 1, 8)=="author: ")
  		highlight.ind <- which(substr(h, 1, 11)=="highlight: ")
		}
		# check for different conditions
		
		if(is.null(title) && exists('t.ind') && length(t.ind)) title <- substr(h[t.ind], 8, nchar(h[t.ind])) else if(is.null(title)) title <- ""
		if(is.null(author) && exists('a.ind') && length(a.ind)) author <- substr(h[a.ind], 9, nchar(h[a.ind])) else if(is.null(author)) author <- ""
		if(is.null(highlight) && exists('highlight.ind') && length(highlight.ind)) highlight <- substr(h[highlight.ind], 12, nchar(h[highlight.ind])) else if(is.null(highlight)) highlight <- hl.default else if(!(highlight %in% knit_theme$get())) highlight <- hl.default
		if(!is.null(title)) header <- c(header, paste0("\\title{", title, "}"))
		if(!is.null(author)) header <- c(header, paste0("\\author{", author, "}"))
		if(!is.null(title)) header <- c(header, "\\maketitle\n")
		
		
		
		header <- c(header, paste0("<<highlight, echo=FALSE>>=\nknit_theme$set(knit_theme$get('", highlight, "'))\n@\n"))
	} else if(ext=="Rnw") {
		from <- rnwChunkID; to <- rmdChunkID
		hl.default <- "tango"
		out.ext <- "Rmd"
	# maketitle is a title page from start to maketitle line	
		if(standalone){
		  begin.doc <- which(l=="\\begin{document}")
		  make.title <- which(l=="\\maketitle")
		} else {begin.doc <- 1; make.title <- 1}
		
		
		if(length(make.title)) h.ind <- 1:make.title else h.ind <- 1:begin.doc
		h <- l[h.ind]
		t.ind <- which(substr(h, 1, 6)=="\\title")
		a.ind <- which(substr(h, 1, 7)=="\\author")
		highlight.ind <- which(substr(l, 1, 11)=="<<highlight")
		if(is.null(title) & length(t.ind)) title <- substr(h[t.ind], 8, nchar(h[t.ind])-1)
		if(is.null(author) & length(a.ind)) author <- substr(h[a.ind], 9, nchar(h[a.ind])-1)
		if(length(highlight.ind)){
			l1 <- l[highlight.ind+1]
			h1 <- substr(l1, nchar("knit_theme$set(knit_theme$get('") + 1, nchar(l1) - nchar("'))\n"))
			if(!(h1 %in% knit_theme$get())) h1 <- hl.default
		}
		# This part is all about headers. H.chunks are header chunks so it must be 
		# evaluated only if this is going to be a standalone document
		
		if(is.null(highlight) & length(highlight.ind)) highlight <- h1 else if(is.null(highlight)) highlight <- hl.default else if(!(highlight %in% knit_theme$get())) highlight <- hl.default
		if(!is.null(standalone) && standalone){
  		header <- .rmdHeader(title = title, author = author, highlight = highlight)
  		h.chunks <- .swapChunks(from = from, to = to, x = h, offset.end = 0)
  		header <- c(header, h.chunks[[1]][h.chunks[[2]]]) %>% paste0(collapse = "\n")
		}
	}
	# l is a main part of the document 

	if(standalone && !is_child) l <- paste0(l[-h.ind], "\n") else l %<>% {paste0(.,"\n")}
	
	l <- .swapHeadings(from = from, to = to, x = l)
	chunks <- .swapChunks(from = from, to = to, x = l)
	l <- chunks[[1]]
	
	if(ext == "Rmd"){ 
	  l <- .swapEmphasis(x = l, emphasis = emphasis)
	  l[-chunks[[2]]] <- sapply(l[-chunks[[2]]], function(v, p, r) gsub(p, r, v), p="_", r="\\\\_")
	}
	if(standalone)   l <- c(header, l)
	if(ext == "Rmd") l <- c(l, "\n\\end{document}\n")
	
	if(ext == "Rnw"){
	  l%<>%.swapItems() %>% .swapEquations()
	  ind <- tryCatch({
	      ret <- which(substr(l, 1, 1) == "\\") 
	      if(length(ret) || !ret) simpleError('Zero substr found!')
	     ret}, error=function(error) return(1))
	   
	   # drop any remaining lines beginning with a backslash
	  l <- l[-ind]
	}
	
	
	
	if(!is.null(is_child) && is_child) outDir <- {if(ext=='Rmd') gsub(ext,'Rnw',x = file) else gsub(ext,'Rmd',x = file)}   %>% {gsub(paste0('\\/',basename(.)),'',.)}
	
	if(!file.exists(outDir)) dir.create(outDir, showWarnings = FALSE)
	outfile <- file.path(outDir, gsub(paste0("\\.", ext), paste0("\\.", out.ext), basename(file)))

				# changing comments 
	if(ext == 'Rmd'){
	  # maybe'^'
	  l %<>% {gsub('<!-- (.*) -->', '%\\1', ., perl=T)}
	} else {
	  l %<>% {gsub('^%(.*)', '<!--\\1 -->', ., perl=T)}
	}
  # Change Rnw childs to Rmd and the opposite, depending on the case	
 
		if(ext == 'Rmd')l %<>% {gsub("\\.Rmd(\'|\")","\\.Rnw\\1",.,perl=T)} else  l %<>% {gsub("\\.Rnw(\'|\")","\\.Rmd\\1",.,perl=T)}
	if(overwrite || !file.exists(outfile)){
	  	  sink(outfile)
	  # for(i in l) cat(i)
	  sapply(l, cat)
	  # writeLines(l,outfile)
		sink(NULL)
		print(paste("Writing", outfile))
	}
}

# .swapItems --------------------------------------------------------------

.swapItems <- function(doc){
  doc%>% 
  {gsub('\\\\begin\\{itemize\\}','',.,perl=T)} %>%
  {gsub('\\\\item','\\*',.,perl=T)} %>% 
  {gsub('\\\\end\\{itemize\\}','',.,perl=T)}
}

# .swapEquations ----------------------------------------------------------

.swapEquations <- function(doc){
  doc %>% 
  {gsub('\\\\begin\\{equation\\}','\\$\\$',.,perl=T)} %>%
  {gsub('\\\\end\\{equation\\}','\\$\\$',.,perl=T)}
}


# convertDocs -------------------------------------------------------------


# @knitr fun_convertDocs
# Rmd <-> Rnw document conversion
# Main conversion function
convertDocs <- function(path, rmdChunkID=c("```{r", "}", "```"), rnwChunkID=c("<<", ">>=", "@"), emphasis="replace", overwrite=FALSE, ...){
	stopifnot(is.character(path))
	type <- basename(path)
	dots <- list(...)
	if(is.null(recursive <- dots$recursive)) recursive <- FALSE
	rmd.files <- list.files(path, pattern=".Rmd$", full=TRUE, recursive=recursive)
	rnw.files <- list.files(path, pattern=".Rnw$", full=TRUE, recursive=recursive)
	
	if(rmdChunkID[1]=="```{r") rmdChunkID[1] <- paste0(rmdChunkID[1], " ")
	if(type=="Rmd"){
		stopifnot(length(rmd.files) > 0)
		outDir <- file.path(dirname(path), "Rnw")
		if(is.null(doc.class <- dots$doc.class)) doc.class <- "article"
		if(is.null(doc.packages <- dots$doc.packages)) doc.packages <- "geometry"
		doc.class.string <- paste0("\\documentclass{", doc.class, "}")
		doc.packages.string <- paste0(sapply(doc.packages, function(x) paste0("\\usepackage{", x, "}")), collapse="\n")
		if("geometry" %in% doc.packages) doc.packages.string <- c(doc.packages.string, "\\geometry{verbose, tmargin=2.5cm, bmargin=2.5cm, lmargin=2.5cm, rmargin=2.5cm}")
		header.rnw <- c(doc.class.string, doc.packages.string, "\\begin{document}\n")#,
			#paste0("<<highlight, echo=FALSE>>=\nknit_theme$set(knit_theme$get('", theme, "'))\n@\n"))
	} else if(type=="Rnw") {
		stopifnot(length(rnw.files) > 0)
	  outDir <- file.path(dirname(path), "Rmd")
	} else stop("path must end in 'Rmd' or 'Rnw'.")
	if(type=="Rmd"){
		sapply(rmd.files, .swap, header=header.rnw, outDir=outDir, rmdChunkID=rmdChunkID, rnwChunkID=rnwChunkID, emphasis=emphasis, overwrite=overwrite, ...)
		cat(".Rmd to .Rnw file conversion complete.\n")
	} else {
		sapply(rnw.files, .swap, header=NULL, outDir=outDir, rmdChunkID=rmdChunkID, rnwChunkID=rnwChunkID, emphasis=emphasis, overwrite=overwrite, ...)
		cat(".Rnw to .Rmd file conversion complete.\n")
	}
}

# moveDocs ----------------------------------------------------------------
# @knitr fun_moveDocs
# Organization documentation
moveDocs <- function(path.docs, type=c("md", "html","pdf"), move=TRUE, copy=FALSE, remove.latex=TRUE, latexDir="latex"){
	if(any(!(type %in% c("md", "html","pdf")))) stop("type must be among 'md', 'html', and 'pdf'")
	stopifnot(move | copy)
	if(path.docs=="." | path.docs=="./") path.docs <- getwd()
	if(strsplit(path.docs, "/")[[1]][1]==".."){
		tmp <- strsplit(path.docs, "/")[[1]][-1]
		if(length(tmp)) path.docs <- file.path(getwd(), paste0(tmp, collapse="/")) else stop("Check path.docs argument.")
	}
	for(i in 1:length(type)){
		if(type[i]=="pdf") origin <- "Rnw" else origin <- "Rmd"
		path.i <- file.path(path.docs, origin)
		infiles <- list.files(path.i, pattern=paste0("\\.", type[i], "$"), full=TRUE)
		if(type[i]=="pdf"){
			extensions <- c("tex", "aux", "log")
			all.pdfs <- basename(list.files(path.docs, pattern=".pdf$", full=T, recursive=T))
			pat <- paste0("^", rep(gsub("pdf", "", all.pdfs), length(extensions)), rep(extensions, each=length(all.pdfs)), "$")
			latex.files <- unlist(sapply(1:length(pat), function(p, path, pat) list.files(path, pattern=pat[p], full=TRUE), path=path.i, pat=pat))
			print(latex.files)
			if(length(latex.files)){
				if(remove.latex){
					unlink(latex.files)
				} else {
					dir.create(file.path(path.docs, latexDir), showWarnings=FALSE, recursive=TRUE)
					file.rename(latex.files, file.path(path.docs, latexDir, basename(latex.files)))
				}
			}
		}
		if(length(infiles)){
			infiles <- infiles[basename(dirname(infiles))==origin]
			if(length(infiles)){
				if(type[i]=="html"){
					html.dirs <- gsub("\\.html", "_files", infiles)
					dirs <- list.dirs(path.i, recursive=FALSE)
					ind <- which(dirs %in% html.dirs)
					if(length(ind)){
						html.dirs <- dirs[ind]
						html.dirs.recur <- list.dirs(html.dirs)
						for(p in 1:length(html.dirs.recur))	dir.create(gsub("/Rmd", "/html", html.dirs.recur[p]), recursive=TRUE, showWarnings=FALSE)
						subfiles <- unique(unlist(lapply(1:length(html.dirs.recur), function(p, path) list.files(path[p], full=TRUE), path=html.dirs.recur)))
						subfiles <- subfiles[!(subfiles %in% html.dirs.recur)]
						file.copy(subfiles, gsub("/Rmd", "/html", subfiles), overwrite=TRUE)
						if(move) unlink(html.dirs, recursive=TRUE)
					}
				}
				outfiles <- file.path(path.docs, type[i], basename(infiles))
				if(move) file.rename(infiles, outfiles) else if(copy) file.copy(infiles, outfiles, overwrite=TRUE)
			}
		}
	}
}
