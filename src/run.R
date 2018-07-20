if(!'pacman' %in% installed.packages()) install.packages('pacman')
library(pacman)
p_load(knitr, tools, magrittr, pander,
       rmarkdown, lubridate, digest, stringi,
       Rcpp, tth, R2HTML, rpm2, assertthat, glue,
       stringr, editR)

TABLE_TYPE = 'html' # 'latex' or 'html'
# Clearing RMD folder

#Converting document

editR::editR(file = './src/chapters/subchapters/3_1_Machine_Learning.Rmd')
editR::editR(file = './src/chapters/subchapters/3_2_reinforcement_learning.Rmd')

#editR::editR(file = './masters_thesis_proposal.Rmd')
input_file  <- 'src/masters_thesis.Rmd'
# input_file  <- 'Rnw/masters_thesis.Rnw'
output_file <- 'masters_thesis.html'

render(input = input_file, output_file = output_file, clean = F);system2('open', args = glue('./{output_file}'), wait = T)






