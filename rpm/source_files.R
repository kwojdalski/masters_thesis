require(plyr)
list.files('./rpm',recursive = T) %>% {grep(pattern ='\\.R$' ,x=., perl=T,value=T)} %>% {paste0('./rpm/',.)} %>% .[c(4)] %>% 
  a_ply(.margins = 1, .fun = function(x){source(x)})





# Some tests --------------------------------------------------------------
path <- './Rnw'
recursive <- 'true'
rmdChunkID=c("```{r", "}", "```")
rnwChunkID=c("<<", ">>=", "@")
emphasis="replace"
overwrite=FALSE
rnw.files <- list.files(path, pattern=".Rnw$", full=TRUE, recursive=recursive)
rnw.files <- rnw.files[1]
outDir <- file.path(dirname(path), "Rmd")
debugonce(.swap)
.swap(rnw.files, header=NULL, outDir=outDir, rmdChunkID=rmdChunkID, rnwChunkID=rnwChunkID, emphasis=emphasis, overwrite=overwrite, standalone=F)
