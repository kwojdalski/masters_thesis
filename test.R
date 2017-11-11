load('swap.RData')
emphasis <- 'replace'
rmdChunkID=c("```{r", "}", "```"); rnwChunkID=c("<<", ">>=", "@")
outDir <- './Rmd'
overwrite <- T


# files -------------------------------------------------------------------
source('./test.R')

file  <- "./Rnw/masters_thesis.Rnw"
file2 <- "./Rnw/chapters/subchapters/1_2_FX_Market_Organization.Rnw"
file3 <- "./Rnw/chapters/subchapters/2.2_Relevant_Financial_Indicators.Rnw"
file4 <- './Rnw/chapters/subchapters/2.1.1_Modern_Portfolio_Theory_and_CAPM.Rnw'
# Convert docs test, with all dependencies and childs, etc
convertDocs('./Rnw',recursive = T)

# Swap test, a whole document test
debugonce(.swap)
  # Rnw -> Rmd
  .swap(file=file, header = header.rnw, outDir=outDir, rmdChunkID=rmdChunkID, rnwChunkID=rnwChunkID, emphasis=emphasis, overwrite=overwrite, standalone=T, child=F)
  .swap(file=file2, header = header.rnw, outDir=outDir, rmdChunkID=rmdChunkID, rnwChunkID=rnwChunkID, emphasis=emphasis, overwrite=overwrite, standalone=F, child=F)
  .swap(file=file4, header = header.rnw, outDir=outDir, rmdChunkID=rmdChunkID, rnwChunkID=rnwChunkID, emphasis=emphasis, overwrite=overwrite, standalone=F, child=F)
  # Rmd -> Rnw
  rnw_to_rmd <- function(file) gsub('Rnw','Rmd',file)
  debugonce(.swap)
  .swap(file=rnw_to_rmd(file), header = NULL, outDir=outDir, rmdChunkID=rmdChunkID, rnwChunkID=rnwChunkID, emphasis=emphasis, overwrite=overwrite, standalone=T, child=F)
  .swap(file=rnw_to_rmd(file2), header = header.rnw, outDir=outDir, rmdChunkID=rmdChunkID, rnwChunkID=rnwChunkID, emphasis=emphasis, overwrite=overwrite, standalone=F, child=F)
  .swap(file=rnw_to_rmd(file4), header = header.rnw, outDir=outDir, rmdChunkID=rmdChunkID, rnwChunkID=rnwChunkID, emphasis=emphasis, overwrite=overwrite, standalone=F, child=F)


# Chunks tests
.swapChunks(from=rnwChunkID, to=rmdChunkID, readLines(file), offset.end=1)[[1]] %>% 
  write(file='aaa.Rmd')

# Swap items test
itemize_file <- readLines(file2)[13:17]
.swapItems(itemize_file)

# Swap equations test
.swapEquations(readLines(file3))
.swap(file=file3, header=header.rnw, outDir=outDir, rmdChunkID=rmdChunkID, rnwChunkID=rnwChunkID, emphasis=emphasis, overwrite=overwrite, standalone=F, child=F)



#knit('./Rmd/masters_thesis.Rmd',output='test.html')

# chapters/subchapters/1_2_FX_Market_Organization.Rmd('|")
#  "'chapters/subchapters/1_2_FX_Market_Organization.Rmd'" %>% {gsub("\\.Rnw(\'|\")","\\.Rmd\1",.,perl=T)}


#
.swapTableType <- function(){
  file_dir <- dirname(sys.frame(1)$ofile) 
  table_type <- if(grepl(file_dir,".Rmd$")) 'html' else 'pdf'
  return(table_type)
}


browser()
.swapTableType <- function(){
  
  file_dir <- dirname(sys.frame(1)$ofile) 
  table_type <- if(grepl(file_dir,".Rmd$")) 'html' else 'pdf'
  return(table_type)
}



