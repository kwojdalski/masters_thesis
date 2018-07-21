if(!'pacman' %in% installed.packages()) install.packages('pacman')
library(pacman)
p_load(knitr, tools, magrittr, pander,
       rmarkdown, lubridate, digest, stringi,
       Rcpp, tth, R2HTML, rpm2, assertthat, glue,
       stringr, editR)

require(shinyAce)
TABLE_TYPE = 'html' # 'latex' or 'html'
# Clearing RMD folder
# options(shiny.error = NULL)
# options(warn = 1)
#Converting document
editR::editR(file = '/Users/krzysztofwojdalski/Dropbox/projekty_r/masters_thesis/src/chapters/3_1_Machine_Learning.Rmd')
options(shiny.error = recover)


#editR::editR(file = './masters_thesis_proposal.Rmd')
input_file  <- './src/masters_thesis.Rmd'

# input_file  <- 'Rnw/masters_thesis.Rnw'
output_file <- 'masters_thesis.pdf'
render(input = input_file, output_file = output_file, clean = T);system2('open', args = glue('./{output_file}'), wait = T)




