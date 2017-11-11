if(!'pacman' %in% installed.packages()) install.packages('pacman')
require(pacman)
p_load(knitr,
       tools,
       magrittr,
       pander,
       rmarkdown,
       lubridate,
       digest,
       stringi,
       Rcpp,
       tth,R2HTML)

source('rpm/code/rpm.R')
opts_knit$set(root.dir = 'H:/Rnw')
file<-'./Rnw/masters_thesis.Rnw'

to_pdf <- function(file, parentcite = T, remove_tmp_files=T) {
  
  file %<>%{gsub('\\.[A-Za-z]*$','',., perl=T)}
  tex_name <- paste0(file,'.tex')
  rnw_name <- paste0(file, '.Rnw')
  Sweave2knitr(rnw_name)
  #tex_tmp_name <- gsub("[.]([^.]+)$", "-knitr.\\1", file)
  tex_tmp_name <- gsub("[.]([^.]+)$", "-knitr.Rnw", rnw_name)
  knit(tex_tmp_name, output = tex_name)
  if (parentcite) 
    readLines(tex_name) %>%
    {gsub(pattern = '\\cite', '\\parencite', .)} %>%
      writeLines(tex_name)
  
  tex <- paste0(file, '.tex')
  tex <<-tex
  texi2pdf(tex, clean=T)
  texi
  # if(remove_tmp_files){
  #   Sys.sleep(3)
  #   list.files() %>% 
  #     .[{grepl(file,.)}]%>% 
  #     .[{!grepl('(*Rmd$)|(.Rnw$)',.)}] %>% 
  #     file.remove()
  # }
  #   
}

debugonce(convertDocs)
# Clearing RMD folder
'./Rmd' %>% {paste(.,list.files(., recursive = T),sep = '/')} %>% file.remove()

#Converting document
convertDocs('./Rmd',recursive = T)
input_file <- './Rmd/masters_thesis.Rmd'
output_file <- './Rmd/masters_thesis.html'
render(input_file, output_file)
debugonce(render)



Sweave(file='masters_thesis.Rnw', driver=RweaveHTML)
ttm(readLines(tex)) %>% 
to_pdf(file)
to_pdf('./chapters/subchapters/1_2_FX_Market_Organization')


system2('open', args = 'masters_thesis.pdf', wait = T)

# Subchapter --------------------------------------------------------------
# # 
# subchapter_dir<-'./chapters/subchapters/1_2_FX_Market_Organization'
# knit(paste0(subchapter_dir,'.Rnw'))
# 
# undebug(knit)
# texi2pdf(paste0(subchapter_dir,'.tex'))
# system2('open', args = './chapters/subchapters/1_2_FX_Market_Organization.pdf', wait = T)
# 
# file.edit('chapters/subchapters/1_2_FX_Market_Organization.tex')
# 
# 
# # readLines('masters_thesis.tex')
# file.edit('masters_thesis.tex')
# 
# undebug(Sweave)
# file.edit('masters_thesis.tex')
# 
# sessionInfo()
# 



