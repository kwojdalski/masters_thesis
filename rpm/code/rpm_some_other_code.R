
# @knitr fun_getProjectStats
# Compile project code and documentation statistics and other metadataR
# Count R scripts, standard functions, lines of code, hierarchical function tree,
# number of Rmd documents, lines of documentation,
# Shiny app reactive expressions not yet included, e.g.,
# input references, output references, render* calls
# other references of interest in the code, e.g., number of conditional panels in a Shiny app
# These instances would be easy to count but a hierarchical reactive elements tree would be challenging
# 'type' argument not currently in use
getProjectStats <- function(path, type=c("project", "app"), code=TRUE, docs=TRUE, exclude=NULL){

	if(!(code | docs)) stop("At least one of 'code' or 'docs' must be TRUE.")
	r.files <- if(code) list.files(path, pattern=".R$", full=TRUE, recursive=TRUE) else NULL
	rmd.files <- if(docs) list.files(path, pattern=".Rmd$", full=TRUE, recursive=TRUE) else NULL
	
	getFunctionInfo <- function(x, func.names=NULL, func.lines=NULL){
		if(is.null(func.names) & is.null(func.lines)){
			x.split <- strsplit(gsub(" ", "", x), "<-function\\(")
			func.ind <- which(sapply(x.split, length) > 1 & !(substr(x, 1, 1) %in% c(" ", "\t")))
			n <- length(func.ind)
			func.names <- if(n > 0) sapply(x.split[func.ind], "[[", 1) else stop("No functions found.")
			func.close <- rep(NA, n)
			for(i in 1:n){
				func.ind2 <- if(i < n) min(func.ind[i+1] - 1, length(x)) else length(x)
				ind <- func.ind[i]:func.ind2
				func.close[i] <- ind[which(nchar(x[ind])==1 & x[ind]=="}")[1]]
			}
			func.lines <- mapply(seq, func.ind, func.close)
			if(!is.list(func.lines)) func.lines <- as.list(data.frame(func.lines))
			return(list(func.names=func.names, func.lines=func.lines, n.func=n))
		} else {
			m <- c()
			n <- length(func.names)
			for(i in 1:n){
				func.ref <- rep(NA, n)
				for(j in c(1:n)[-i]){
					x.tmp <- x[func.lines[[i]]]
					x.tmp <- gsub(paste0(func.names[j], "\\("), "_1_SOMETHING_TO_SPLIT_ON_2_", x.tmp) # standard function usage
					x.tmp <- gsub(paste0("do.call\\(", func.names[j]), "_1_SOMETHING_TO_SPLIT_ON_2_", x.tmp) # function reference inside do.call()
					x.tmp <- gsub(paste0(func.names[j], ","), "_1_SOMETHING_TO_SPLIT_ON_2_", x.tmp) # function reference followed by mere comma, e.g., in *apply functions: NOT IDEAL
					x.tmp.split <- strsplit(x.tmp, "SOMETHING_TO_SPLIT_ON")
					func.ref[j] <- any(sapply(x.tmp.split, length) > 1)
				}
				m.tmp <- if(any(func.ref, na.rm=TRUE)) cbind(func.names[i], func.names[which(func.ref)]) else cbind(func.names[i], NA)
				m <- rbind(m, m.tmp)
			}
			return(flow=m)
		}
	}
	
	if(is.character(exclude) & length(r.files)) r.files <- r.files[!(basename(r.files) %in% exclude)]
	n.scripts <- length(r.files)
	if(n.scripts > 0){
		l <- unlist(lapply(r.files, readLines))
		n.codelines <- length(l[l != ""])
		func.info <- getFunctionInfo(l)
		func.names <- func.info$func.names
		n.func <- func.info$n.func
		func.mat <- getFunctionInfo(l, func.names=func.names, func.lines=func.info$func.lines)
	} else { n.codelines <- n.func <- 0; func.names <- func.mat <- NULL }
	
	if(is.character(exclude) & length(rmd.files)) rmd.files <- rmd.files[!(basename(r.files) %in% exclude)]
	n.docs <- length(rmd.files)
	if(n.docs > 0){
		l <- unlist(lapply(rmd.files, readLines))
		n.doclines <- length(l[l != ""])
	} else { n.doclines <- 0 }
	
	total.files <- length(list.files(path, recursive=TRUE))	
	
	return(list(total.files=total.files, n.docs=n.docs, n.doclines=n.doclines, n.scripts=n.scripts, n.codelines=n.codelines, n.func=n.func, func.mat=func.mat))
}

# @knitr fun_buttonGroup
# Functions for Github websites
buttonGroup <- function(txt, urls, fa.icons=NULL, colors="primary", solid.group=FALSE){
	stopifnot(is.character(txt) & is.character(urls))
	n <- length(txt)
	stopifnot(length(urls)==n)
	stopifnot(colors %in% c("default", "primary", "success", "info", "warning", "danger", "link"))
	stopifnot(n %% length(colors)==0)
	if(is.null(fa.icons)) icons <- vector("list", length(txt)) else if(is.character(fa.icons)) icons <- as.list(fa.icons) else stop("fa.icons must be character or NULL")
	stopifnot(length(icons)==n)
	if(length(colors) < n) colors <- rep(colors, length=n)
	
	btnlink <- function(i, txt, url, icon, col){
		x <- paste0('<a class="btn btn-', col[i], '" href="', url[i], '">')
		y <- if(is.null(icon[[i]])) "" else paste0('<i class="fa fa-', icon[[i]], ' fa-lg"></i>')
		z <- paste0(" ", txt[i], '</a>\n')
		paste0(x, y, z)
	}
	
	x <- if(solid.group) '<div class="btn-group btn-group-justified">\n' else ""
	y <- paste0(sapply(1:length(txt), btnlink, txt=txt, url=urls, icon=icons, col=colors), collapse="")
	z <- if(solid.group) '</div>\n' else ""
	paste0(x, y, z)
}

# @knitr fun_genNavbar
genNavbar <- function(htmlfile="navbar.html", before_body=NULL, title, menu, submenus, files, title.url="index.html", home.url="index.html", site.url="", site.name="Github", media.button.args=NULL, include.home=FALSE){
	ncs <- c("navbar-brand", "navbar-collapse collapse navbar-responsive-collapse", "nav navbar-nav", "nav navbar-nav navbar-right", "container", "navbar-header", "      </div>\n", "navbar-toggle", ".navbar-responsive-collapse", "")
	
	if(!is.null(media.button.args)){
		media.buttons <- do.call(buttonGroup, media.button.args)
	} else if(site.name=="Github" & site.url!="") {
		media.buttons <- paste0('<a class="btn btn-link" href="', site.url, '">\n            <i class="fa fa-github fa-lg"></i>\n            ',site.name,'\n          </a>\n')
	} else media.buttons <- ""
	
	fillSubmenu <- function(x, name, file){
		dd.menu.header <- "dropdown-header"
		if(file[x]=="divider") return('              <li class="divider"></li>\n')
		if(file[x]=="header") return(paste0('              <li class="', dd.menu.header, '">', name[x], '</li>\n'))
		paste0('              <li><a href="', file[x], '">', name[x], '</a></li>\n')
	}
	
	fillMenu <- function(x, menu, submenus, files){
		m <- menu[x]
		gs.menu <- gsub(" ", "-", tolower(m))
		s <- submenus[[x]]
		f <- files[[x]]
		if(s[1]=="empty"){
			y <- paste0('<li><a href="', f,'">', m, '</a></li>\n')
		} else {
			y <- paste0(
			'<li class="dropdown">\n            <a href="', 
				gs.menu, 
				'" class="dropdown-toggle" data-toggle="dropdown">', m, 
				' <b class="caret"></b></a>\n            <ul class="dropdown-menu">\n',
				paste(sapply(1:length(s), fillSubmenu, name=s, file=f), sep="", collapse=""),
				'            </ul>\n', collapse="")
		}
	}
	
	if(include.home) home <- paste0('<li><a href="', home.url, '">Home</a></li>\n          ') else home <- ""
	x <- paste0(
		'<div class="navbar navbar-default navbar-fixed-top">\n  <div class="', ncs[5], '">\n    <div class="', ncs[6], '">\n      <button type="button" class="', ncs[8], '" data-toggle="collapse" data-target="', ncs[9], '">\n        <span class="icon-bar"></span>\n        <span class="icon-bar"></span>\n        <span class="icon-bar"></span>\n      </button>\n      <a class="', ncs[1], '" href="', title.url, '">', title, '</a>\n', ncs[7], '      <div class="', ncs[2], '">\n        <ul class="', ncs[3], '">\n          ',
		home,
		paste(sapply(1:length(menu), fillMenu, menu=menu, submenus=submenus, files=files), sep="", collapse="\n          "),
		'        </ul>\n        <ul class="', ncs[4], '">\n          ', media.buttons, '        </ul>\n      </div><!--/.nav-collapse -->\n    </div>\n  ', ncs[10], '</div>\n',
		collpase="")
		
	if(!is.null(before_body)) x <- paste0(readLines(before_body), x)
	
	sink(htmlfile)
	cat(x)
	sink()
	x
}

# @knitr fun_genOutyaml
genOutyaml <- function(file, theme="cosmo", highlight="zenburn", lib=NULL, header=NULL, before_body=NULL, after_body=NULL){
	output.yaml <- paste0('html_document:\n  self_contained: false\n  theme: ', theme, '\n  highlight: ', highlight, '\n  mathjax: null\n  toc_depth: 2\n')
	if(!is.null(lib)) output.yaml <- paste0(output.yaml, '  lib_dir: ', lib, '\n')
	output.yaml <- paste0(output.yaml, '  includes:\n')
	if(!is.null(header)) output.yaml <- paste0(output.yaml, '    in_header: ', header, '\n')
	if(!is.null(before_body)) output.yaml <- paste0(output.yaml, '    before_body: ', before_body, '\n')
	if(!is.null(after_body)) output.yaml <- paste0(output.yaml, '    after_body: ', after_body, '\n')
	sink(file)
	cat(output.yaml)
	sink()
	output.yaml
}

# @knitr fun_insert_gatc
insert_gatc <- function(file, gatc=NULL){
	nc <- nchar(file)
	stopifnot(all(substr(file, nc-4, nc)==".html"))
	l <- lapply(file, readLines)
	l.ind <- sapply(l, function(x) which(gsub(" ", "", substr(x, 1, 7)) == "</head>"))
	if(is.null(gatc)) gatc <-
"<script>
  (function(i,s,o,g,r,a,m){i['GoogleAnalyticsObject']=r;i[r]=i[r]||function(){
  (i[r].q=i[r].q||[]).push(arguments)},i[r].l=1*new Date();a=s.createElement(o),
  m=s.getElementsByTagName(o)[0];a.async=1;a.src=g;m.parentNode.insertBefore(a,m)
  })(window,document,'script','//www.google-analytics.com/analytics.js','ga');

  ga('create', 'UA-46129458-3', 'auto');
  ga('send', 'pageview');

</script>\n
"
	l <- lapply(1:length(l), function(i, x, ind, gatc) { x[[i]][ind[i]] <- paste0(gatc, "\n</head>"); x[[i]] }, x=l, ind=l.ind, gatc=gatc)
	lapply(1:length(file),
		function(i, file, x){
			sink(file[i])
			x <- x[[i]]
			x <- paste0(x, "\n")
			sapply(x, cat)
			sink()
		}, file=file, x=l)
	
	cat("Google Analytics tracking script inserted.\n")
}

# @knitr fun_genPanelDiv
genPanelDiv <- function(outDir, type="projects", main="Projects",
	github.user="leonawicz", prjs.dir="C:/github", exclude=c("leonawicz.github.io", "shiny-apps", "eris2", "DataVisExamples", ".git", "_images"),
	img.loc="_images/small", lightbox=FALSE, include.buttons=TRUE, include.titles=TRUE, ...){
	
	stopifnot(github.user %in% c("leonawicz", "ua-snap"))
	dots <- list(...)
	apps.append <- list(...)$apps.container.append
    if(type!="apps" || !is.logical(apps.append)) apps.append <- FALSE
    apps.subset <- list(...)$apps.subset
    
	if(type=="apps"){
		filename <- "apps_container.html"
		web.url <- "http://shiny.snap.uaf.edu"
		gh.url.tail <- "shiny-apps/tree/master"
		atts <- ' target="_blank"'
		go.label <- "Launch"
		prjs.dir <- file.path(prjs.dir, "shiny-apps")
		prjs.img <- list.files(file.path(prjs.dir, img.loc))
		prjs <- sapply(strsplit(prjs.img, "\\."), "[[", 1)
        prjs.ind <- if(length(apps.subset)) match(apps.subset, prjs) else seq_along(prjs)
        prjs <- prjs[prjs.ind]
        prjs.img <- prjs.img[prjs.ind]
	}
	if(type=="projects"){
		filename <- "projects_container.html"
		web.url <- paste0("http://", github.user, ".github.io")
		gh.url.tail <- ""
		atts <- ""
		go.label <- "Website"
		prjs <- list.dirs(prjs.dir, full=TRUE, recursive=FALSE)
		prjs <- prjs[!(basename(prjs) %in% exclude)]
		prjs.img <- sapply(1:length(prjs), function(i, a) list.files(file.path(a[i], "plots"), pattern=paste0("^_", basename(a)[i])), a=prjs)
		prjs <- basename(prjs)
	}
	if(type=="datavis"){
		filename <- "data-visualizations_container.html"
		web.url <- paste0("http://", github.user, ".github.io")
		gh.url.tail <- "DataVisExamples/tree/master"
		atts <- ""
		go.label <- "See More"
		prjs.dir <- file.path(prjs.dir, "DataVisExamples")
		prjs.img <- list.files(file.path(prjs.dir, img.loc))
		prjs <- sapply(strsplit(prjs.img, "\\."), "[[", 1)
	}
	if(type=="gallery"){
		web.url <- paste0("http://", github.user, ".github.io")
		gh.url.tail <- "DataVisExamples/tree/master"
		if(lightbox) atts1 <- ' data-lightbox="ID"' else atts1 <- ""
		go.label <- "Expand"
		prjs <- list.dirs(file.path(prjs.dir, "DataVisExamples"), full=T, recursive=F)
		prjs <- prjs[!(basename(prjs) %in% exclude)]
		prjs.img <- lapply(1:length(prjs), function(x, files, imgDir) list.files(path=file.path(files[x], imgDir), recursive=FALSE), files=prjs, imgDir=img.loc)
		prjs <- basename(prjs)
		filename <- tolower(paste0("gallery-", gsub(" ", "-", gsub(" - ", " ", prjs)), ".html"))
	}
	gh.url <- file.path("https://github.com", github.user, gh.url.tail)
	
    fillRow <- function(i, ...){
		prj <- panels[i]
		go.label <- go.label[i]
		col <- col[i]
	    panel.main <- panel.main[i]
		if(type=="apps") img.src <- file.path(gsub("/tree/", "/raw/", gh.url), img.loc, prjs.img[i])
		if(type=="projects") img.src <- file.path(gh.url, prj, "raw/master/plots", prjs.img[i])
		if(type=="datavis") img.src <- file.path(gsub("/tree/", "/raw/", gh.url), img.loc, prjs.img[i])
	    if(type!="gallery"){
			if(type=="datavis"){
				pfx <- "gallery-"
				sfx <- ".html"
				base <- tolower(paste0(pfx, gsub("_", "-", gsub("_-_", "-", prj)), sfx))
			} else {
				base <- prj
			}
			web.url <- file.path(web.url, base)
		} else {
			prj <- prjs[p]
			img.src <- file.path(gsub("/tree/", "/raw/", gh.url), prjs[p], img.loc, panels[i])
			web.url <- file.path(gsub("/tree/", "/raw/", gh.url), prjs[p], panels[i])
			if(lightbox) atts <- gsub("ID", gsub(" - ", ": ", gsub("_", " ", prjs[p])), atts1) else atts <- atts1
		}
		if(include.titles){
		panel.title <- paste0('<div class="panel-heading"><h3 class="panel-title">', panel.main, '</h3>
          </div>\n          ')
		} else panel.title <- ""
		if(include.buttons){
			if(go.label=="UAF ONLY") { web.url <- "#"; atts <- ""; go.btn <- "danger" } else go.btn <- "success"
			panel.buttons <- paste0('<div class="btn-group btn-group-justified">
			<a href="', web.url, '"', atts, ' class="btn btn-', go.btn, '">', go.label, '</a>
			<a href="', file.path(gh.url, prj), '" class="btn btn-info">Github</a>
		  </div>\n        ')
		} else panel.buttons <- ""
	    x <- paste0('    <div class="col-lg-4">
      <div class="bs-component">
        <div class="panel panel-', col, '">\n          ', panel.title,
         '<div class="panel-body"><a href="', web.url, '"', atts, '><img src="', img.src, '" alt="', panel.main, '" width=100% height=200px></a><p></p>\n          ', panel.buttons,
		 '  </div>\n        </div>\n      </div>\n    </div>\n  ')
	}
	
	for(p in 1:length(filename)){
		if(type=="gallery"){
			panels <- prjs.img[[p]]
			main <- gsub(" - ", ": ", gsub("_", " ", prjs[p]))
		} else panels <- prjs
		n <- length(panels)
		if(is.null(dots$go.label)) go.label <- rep(go.label, length=n) else go.label <- rep(dots$go.label, length=n)
		if(is.null(dots$col)) col <- rep("warning", length=n) else col <- rep(dots$col, length=n)
		if(is.null(dots$panel.main)) panel.main <- gsub(" - ", ": ", gsub("_", " ", panels)) else panel.main <- rep(dots$panel.main, length=n)
		seq1 <- seq(1, n, by=3)
		x <- paste0('<div class="container">\n  <div class="row">\n    <div class="col-lg-12">\n      <div class="page-header">\n        <h3 id="', type, '">', main, '</h3>\n      </div>\n    </div>\n  </div>\n  ')
		y <- c()
		for(j in 1:length(seq1)){
			ind <- seq1[j]:(seq1[j] + 2)
			ind <- ind[ind %in% 1:n]
			y <- c(y, paste0('<div class="row">\n', paste0(sapply(ind, fillRow, panels=panels, go.label=go.label, col=col, panel.main=panel.main), collapse="\n"), '</div>\n'))
		}
		z <- '</div>\n'
		sink(file.path(outDir, filename[p]), append=apps.append)
		sapply(c(x, y, z), cat)
		sink()
		cat("div container html file created.\n")
	}
}

# @knitr fun_htmlHead
htmlHead <- function(author="Matthew Leonawicz", title=author, script.paths=NULL, stylesheet.paths, stylesheet.args=vector("list", length(path.stylesheets)), include.ga=TRUE, ...){
x <- paste0('<!DOCTYPE html>

<html xmlns="http://www.w3.org/1999/xhtml">

<head>

<meta charset="utf-8">
<meta http-equiv="Content-Type" content="text/html; charset=utf-8" />

<meta name="author" content=', author, ' />

<title>', title, '</title>
')

if(is.character(script.paths)) x <- c(x, paste0(paste0('<script src="', script.paths, '"></script>', collapse="\n"), "\n"))

x <- c(x, '<meta name="viewport" content="width=device-width, initial-scale=1.0" />\n')

if(is.character(stylesheet.paths)){
	n <- length(stylesheet.paths)
	stopifnot(is.list(stylesheet.args))
	stopifnot(length(stylesheet.args)==n)
	for(i in 1:n){
		string <- ""
		if(is.list(stylesheet.args[i])){
			v <- stylesheet.args[i]
			arg <- names(v)
			if(is.character(arg) && all(arg!="")) string <- paste0(" ", paste(arg, paste0('\"', v, '\"'), sep="=", collapse=" "))
		}
		x <- c(x, paste0('<link rel="stylesheet" href="', stylesheet.paths[i], '"', string, '>\n'))
	}
}

if(include.ga) {
x <- c(x,
"<script>
  (function(i,s,o,g,r,a,m){i['GoogleAnalyticsObject']=r;i[r]=i[r]||function(){
  (i[r].q=i[r].q||[]).push(arguments)},i[r].l=1*new Date();a=s.createElement(o),
  m=s.getElementsByTagName(o)[0];a.async=1;a.src=g;m.parentNode.insertBefore(a,m)
  })(window,document,'script','//www.google-analytics.com/analytics.js','ga');

  ga('create', 'UA-46129458-3', 'auto');
  ga('send', 'pageview');

</script>\n
")
}

x <- c(x, '</head>\n')
x

}

# @knitr fun_htmlBodyTop
htmlBodyTop <- function(css.file=NULL, css.string=NULL, background.image="", include.default=TRUE, ...){
	x <- '<body>\n<style type = "text/css">\n'
	
	default <- paste0('
	.main-container {
	  max-width: 940px;
	  margin-left: auto;
	  margin-right: auto;

	}

	body {
	  background-image: url("', background.image, '");
	  background-attachment: fixed;
	  background-size: 1920px 1080px;
	}
	
	/* padding for bootstrap navbar */
	body {
	  padding-top: 50px;
	  padding-bottom: 40px;
	}
	@media (max-width: 979px) {
	  body {
		padding-top: 0;
	  }
	}
	
	.nav>.btn {
	  line-height: 0.75em;
	  margin-top: 9px;
	}
	')
	
	if(!is.null(css.file)) y <- readLines(css.file) else y <- ""
	if(!is.null(css.string)) y <- c(y, css.string)
	if(include.default) y <- c(default, y)
	
	z <- '\n</style>\n'

	c(x, y, z)
}

# @knitr fun_htmlBottom
htmlBottom <- function(...){ # temporary
	'<div class="container">Site made with <a href="http://leonawicz.github.io/ProjectManagement">rpm</a></div>\n</body>\n</html>'
}


# @knitr fun_genUserPage
genUserPage <- function(file="C:/github/leonawicz.github.io/index.html", containers=NULL, navbar="", ...){
	x1 <- htmlHead(...)
	x2 <- htmlBodyTop(...)
	if(!is.null(containers)) x3 <- sapply(containers, function(x) paste0(paste0(readLines(x), collapse="\n"), "\n\n")) else x3 <- ""
	x4 <- htmlBottom(...)
	nb <- if(file.exists(navbar) && substr(navbar, nchar(navbar)-4, nchar(navbar))==".html") nb <- paste0(paste0(readLines(navbar), collapse="\n"), "\n\n")
	sink(file)
	sapply(c(x1, x2, nb, x3, x4), cat)
	sink()
	cat("Github User page html file created.\n")
}
