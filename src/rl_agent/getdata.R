GetSingle <- function(ch_assetCode = "", ch_dataDir = "", ch_fileFormat = ""){  ##ch_dataType, ## get's data from file for a single asset

#   if(!file.exists(paste(ch_dataDir,ch_assetCode,".txt",sep=""))) ## checks if the file exists and creates it if not
#   {
#     SaveData(ch_assetCode, ch_dataDir)
#   }
  df_assetData <- read.table(file.path(ch_dataDir, paste0(ch_assetCode, ch_fileFormat)), header=TRUE, sep="\t", 
                            fill=FALSE, strip.white=TRUE)#, col.names=paste("price_", ch_assetCode, sep=""))
  #df_assetData <- df_assetData[,'avgaskp', drop = FALSE]
  
  return(df_assetData)
}

#EURGBP <- GetSingle(c("EURGBP"), "Data//Time Series//1 day 201203050000//", ".csv")  ## checks if data is accesible for a single asset
#asset <- GetSingle(c("AUDCADask"), "C:\\Users\\user\\Desktop\\ATRG\\data\\data\\", ".csv")
#asset <- GetSingle(c("EURGBPask"), "C:\\Users\\user\\Desktop\\ATRG\\data\\data\\", ".csv")
asset <- GetSingle(c("EURGBP"), data_path, ".csv")
GetQuotes <- function(c_assets, ch_dataDir = "", ch_fileFormat ="") ## creates a list of quotations for the given asset names vector.
{
  
  df_quotes <- data.frame(matrix(0, nrow = nrow(GetSingle(c_assets[1], ch_dataDir, ch_fileFormat)), ncol = length(c_assets)))  ## prealocation of space
  colnames(df_quotes) <- c_assets
  
  
  for (ch_asset in c_assets)
  { 
    df_asset <- GetSingle(ch_asset, ch_dataDir, ch_fileFormat)
    df_quotes[[ch_asset]] <- df_asset[,1]                          
  }  
 
  return(df_quotes)
}

randomObs <- function(vector = character(), n){
  return(sample(vector,n))
}

randomRows <- function(df,n){
  return(df[sample(nrow(df),n),])
}

#c_currencies <- c("AUDCAD", "AUDHKD")
#df_currencies <- GetQuotes(c_currencies, "Data//Time Series//")