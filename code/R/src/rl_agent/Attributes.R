# An Reinforcement Learning Application for Statistical Arbitrage
# karol.przybylak.13@ucl.ac.uk, kmprzybylak@gmail.com

# Attributes generation

# -----------------------------------------------------------------
#region Dependencies

library(TTR)


#endregion
# -----------------------------------------------------------------
#region Auxiliary Functions

ReturnOnInvestment <- function(l_quotes, i_period) {
  c_differences <- diff(l_quotes, i_period)
  returnOnInvestment <- c(
    (1:i_period) * NA,
    c_differences / l_quotes[1:length(c_differences)]
  ) +
    1
  return(returnOnInvestment)
}

MinMax <- function(l_QuotesHigh, l_QuotesLow) {
  return(100 * (l_QuotesHigh - l_QuotesLow) / l_QuotesLow)
}


#endregion
# -----------------------------------------------------------------
#region Attribute functions

AttributesAll <- function(
  df_currencies,
  c_currencies,
  i_delay,
  base_series = list('price', 'return')
) {
  #l_attributes <- list()
  l_attributes <- plyr::alply(c_currencies, 1, function(asset) {
    Attributes(asset, df_currencies[asset], i_delay, FALSE)
  })
  return(l_attributes)
}


Attributes <- function(c_assets, l_quotes, i_delay, lo_naOmit) {
  df_attributes <- createAttributes(c_assets, l_quotes, i_delay)
  #df_attributes <- CorrectAttributes(df_attributes, l_quotes, i_delay, lo_naOmit)

  return(df_attributes)
}

createAttributes <- function(c_assets, l_quotes, i_delay) {
  df_attributes <- data.frame(row.names = c(1:length(l_quotes[[1]])))

  for (ch_asset in c_assets) {
    i_return <- ReturnOnInvestment(l_quotes[[ch_asset]], i_delay)
    #5-dniowy zwrot z inwestycji
    df_attributes[paste("zwrot_", ch_asset, sep = "")] <- i_return
    #5-dniowa ?rednia ruchoma atrybutu zwrot
    #df_attributes[paste("MA3_zwrot_", ch_asset, sep = "")] <- SMA(i_return, n = 3)
    #30-dniowa ?rednia ruchoma atrybutu zwrot
    #df_attributes[paste("MA15_zwrot_", ch_asset, sep = "")] <- SMA(i_return, n = 15)
    #Wyk?adnicza 5-dniowa ?rednia ruchoma atrybutu zwrot
    df_attributes[paste("EMA3_zwrot_", ch_asset, sep = "")] <- EMA(
      i_return,
      n = 3,
      wilder = TRUE
    )
    #Wyk?adnicza 30-dniowa ?rednia ruchoma atrybutu zwrot
    #df_attributes[paste("EMA15_zwrot_", ch_asset, sep = "")] <- EMA(i_return, n = 15, wilder = TRUE)
    #Wyk?adnicza 60-dniowa ?rednia ruchoma atrybutu zwrot
    #df_attributes[paste("EMA60_zwrot_", ch_asset, sep = "")] <- EMA(i_return, n = 60, wilder = TRUE)
    #Wyk?adnicza 150-dniowa ?rednia ruchoma atrybutu zwrot
    df_attributes[paste("EMA10_zwrot_", ch_asset, sep = "")] <- EMA(
      i_return,
      n = 10,
      wilder = TRUE
    )
    #Wa?ona 5-dniowa ?rednia ruchoma atrybutu zwrot
    #df_attributes[paste("WMA3_zwrot_", ch_asset, sep = "")] <- WMA(i_return, n = 3)
    #Wa?ona 30-dniowa ?rednia ruchoma atrybutu zwrot
    #df_attributes[paste("WMA15_zwrot_", ch_asset, sep = "")] <- WMA(i_return, n = 15)
    #Wa?ona 60-dniowa ?rednia ruchoma atrybutu zwrot
    #df_attributes[paste("WMA60_zwrot_", ch_asset, sep = "")] <- WMA(i_return, n = 60)
    #Warto?? wsp??czynnika beta dla danej sp??ki
    #df_attributes[paste("beta_", ch_asset, sep = "")]
    #____________________________________________________________________________________________________________

    #MACD indicator
    #macd <-  MACD(l_quotes[[ch_asset]], 12, 26, 9, maType="EMA")
    #df_attributes[paste("MACD_", ch_asset, sep = "")]        <- macd[1:nrow(df_attributes),1]
    #df_attributes[paste("MACDsigLine_", ch_asset, sep = "")] <- macd[1:nrow(df_attributes),2]
    #df_attributes[paste("MACDsignal_", ch_asset, sep = "")]  <- macd[1:nrow(df_attributes),1] - macd[1:nrow(df_attributes),2]
    #____________________________________________________________________________________________________________

    #TEMA indicator
    kurs <- l_quotes[[ch_asset]]
    #TEMA15 dniowa
    Tema3 <- 3 *
      EMA(kurs, n = 3) -
      (3 * (EMA(EMA(kurs, n = 3), n = 3))) +
      EMA(EMA(EMA(kurs, n = 3), n = 3), n = 3)
    df_attributes[paste("TEMA3_", ch_asset, sep = "")] <- Tema3[
      1:nrow(df_attributes)
    ]
    #TEMA60dniowa
    #Tema60 <- 3*EMA(kurs, n=60) - (3*(EMA(EMA(kurs, n=60), n=60))) + EMA(EMA(EMA(kurs, n=60), n=60), n=60)
    #df_attributes[paste("TEMA60_", ch_asset, sep = "")] <- Tema60[1:nrow(df_attributes)]
    #TEMA30dniowa wilder
    #Tema30w <- 3*EMA(kurs, n=30, wilder = TRUE) - (3*(EMA(EMA(kurs, n=30, wilder = TRUE), n=30, wilder = TRUE))) + EMA(EMA(EMA(kurs, n=30, wilder = TRUE), n=30, wilder = TRUE), n=30, wilder = TRUE)
    #df_attributes[paste("TEMA30w_", ch_asset, sep = "")] <- Tema30w[1:nrow(df_attributes)]

    #TEMA150dniowa wilder
    #Tema10 <- 3*EMA(kurs, n=100, wilder = TRUE) - (3*(EMA(EMA(kurs, n=10, wilder = TRUE), n=10, wilder = TRUE))) + EMA(EMA(EMA(kurs, n=10, wilder = TRUE), n=10, wilder = TRUE), n=10, wilder = TRUE)
    #df_attributes[paste("TEMA10_", ch_asset, sep = "")] <- Tema10[1:nrow(df_attributes)]
    #____________________________________________________________________________________________________________

    #AROON index 8 days
    #aroon8 <- aroon(l_quotes[[ch_asset]]$High, n = 8)
    #aroon8 <- aroon(l_quotes[[ch_asset]], n = 8)
    #df_attributes[paste("AROON8up_", ch_asset, sep = "")] <- aroon8[1:nrow(df_attributes),1]
    #df_attributes[paste("AROON8dn_", ch_asset, sep = "")] <- aroon8[1:nrow(df_attributes),2]
    #df_attributes[paste("AROON8osc_", ch_asset, sep = "")] <- aroon8[1:nrow(df_attributes),3]
    #AROON index 15 days
    #aroon15 <- aroon(l_quotes[[ch_asset]]$High, n = 15)
    #df_attributes[paste("AROON15up_", ch_asset, sep = "")] <- aroon15[1:nrow(df_attributes),1]
    #df_attributes[paste("AROON15dn_", ch_asset, sep = "")] <- aroon15[1:nrow(df_attributes),2]
    #df_attributes[paste("AROON15osc_", ch_asset, sep = "")] <- aroon15[1:nrow(df_attributes),3]
    #AROON index 60 days
    #aroon60 <- aroon(l_quotes[[ch_asset]]$High, n = 60)
    #df_attributes[paste("AROON60up_", ch_asset, sep = "")] <- aroon60[1:nrow(df_attributes),1]
    #df_attributes[paste("AROON60dn_", ch_asset, sep = "")] <- aroon60[1:nrow(df_attributes),2]
    #df_attributes[paste("AROON60osc_", ch_asset, sep = "")] <- aroon60[1:nrow(df_attributes),3]
    #AROON index 150 days
    #aroon100 <- aroon(l_quotes[[ch_asset]], n = 100)
    #df_attributes[paste("AROON100up_", ch_asset, sep = "")] <- aroon150[1:nrow(df_attributes),1]
    #df_attributes[paste("AROON100dn_", ch_asset, sep = "")] <- aroon150[1:nrow(df_attributes),2]
    #df_attributes[paste("AROON100osc_", ch_asset, sep = "")] <- aroon100[1:nrow(df_attributes),3]
    #___________________________________________________________________________________________________________

    #ATR index
    #HLC <- data.frame(l_quotes[[ch_asset]]$High, l_quotes[[ch_asset]]$Low, l_quotes[[ch_asset]]$Close)
    #ATR 14 days
    #ATR8 <- ATR(HLC, maType="EMA", n=8)
    #df_attributes[paste("ATR8tr_", ch_asset, sep = "")] <- ATR8[1:nrow(df_attributes),1]
    #df_attributes[paste("ATR8atr_", ch_asset, sep = "")] <- ATR8[1:nrow(df_attributes),2]
    #df_attributes[paste("ATR8Hi_", ch_asset, sep = "")] <- ATR8[1:nrow(df_attributes),3]
    #df_attributes[paste("ATR8lo_", ch_asset, sep = "")] <- ATR8[1:nrow(df_attributes),4]
    #ATR 60 days
    #ATR60 <- ATR(HLC, maType="EMA", n=60)
    #df_attributes[paste("ATR60tr_", ch_asset, sep = "")] <- ATR60[1:nrow(df_attributes),1]
    #df_attributes[paste("ATR60atr_", ch_asset, sep = "")] <- ATR60[1:nrow(df_attributes),2]
    #df_attributes[paste("ATR60Hi_", ch_asset, sep = "")] <- ATR60[1:nrow(df_attributes),3]
    #df_attributes[paste("ATR60lo_", ch_asset, sep = "")] <- ATR60[1:nrow(df_attributes),4]
    #ATR 100 days
    #ATR100 <- ATR(HLC, maType="EMA", n=100)
    #df_attributes[paste("ATR100tr_", ch_asset, sep = "")] <- ATR100[1:nrow(df_attributes),1]
    #df_attributes[paste("ATR100atr_", ch_asset, sep = "")] <- ATR100[1:nrow(df_attributes),2]
    #df_attributes[paste("ATR100Hi_", ch_asset, sep = "")] <- ATR100[1:nrow(df_attributes),3]
    #df_attributes[paste("ATR100lo_", ch_asset, sep = "")] <- ATR100[1:nrow(df_attributes),4])
    #____________________________________________________________________________________________________________

    # Bolinger Bands
    #BBands20 <- BBands(HLC, n=20, maType = "SMA", sd=2)
    #df_attributes[paste("BBands20dn_", ch_asset, sep = "")] <- BBands20[1:nrow(df_attributes),1]
    #df_attributes[paste("BBands20mavg_", ch_asset, sep = "")] <- BBands20[1:nrow(df_attributes),2]
    #df_attributes[paste("BBands20up_", ch_asset, sep = "")] <- BBands20[1:nrow(df_attributes),3]
    #df_attributes[paste("BBands20pctB_", ch_asset, sep = "")] <- BBands20[1:nrow(df_attributes),4]
    #df_attributes[paste("BBands20Bdiv_", ch_asset, sep = "")]  <- BBands20[1:nrow(df_attributes),3] - BBands20[1:nrow(df_attributes),1]
    #df_attributes[paste("BBands20Pdiv_", ch_asset, sep = "")]  <- l_quotes[[ch_asset]]$Close - BBands20[1:nrow(df_attributes),2]

    #____________________________________________________________________________________________________________

    # Chaikin Accumulation / Distribution line
    #df_attributes[paste("ChaikinAD_", ch_asset, sep = "")] <- chaikinAD(HLC, l_quotes[[ch_asset]]$Volume)
    #df_attributes[paste("ChaikinMoneyFlow_", ch_asset, sep = "")] <-  (ReturnOnInvestment(l_quotes[[ch_asset]]$Close, 10) - 1) * chaikinAD(HLC, l_quotes[[ch_asset]]$Volume)  #uzaleznic jakos od sredniej wartosci wystepowania seri..(jak dlugie srednio sa)

    #Chaikin VOlatility index 10 days
    #HighLow <- data.frame(l_quotes[[ch_asset]]$High,l_quotes[[ch_asset]]$Low)
    #df_attributes[paste("ChaikinVOLA_", ch_asset, sep = "")] <- chaikinVolatility(HighLow, n=8, maType = "SMA")
    #____________________________________________________________________________________________________________

    #wolumen  Ilo?? akcji danego waloru, kt?re zmieni?y w?a?ciciela
    #df_attributes[paste("wolumen_", ch_asset, sep = "")] <- l_quotes[[ch_asset]]$Volume
    #5- dniowa ?rednia ruchoma dla wolumenu
    #df_attributes[paste("MA8_wolumen_", ch_asset, sep = "")] <- SMA(l_quotes[[ch_asset]]$Volume, n = 8)
    #30 - dniowa ?rednia ruchoma dla wolumenu
    #df_attributes[paste("MA15_wolumen_", ch_asset, sep = "")] <- SMA(l_quotes[[ch_asset]]$Volume, n = 15)
    #30 - dniowa ?rednia ruchoma dla wolumenu
    #df_attributes[paste("MA60_wolumen_", ch_asset, sep = "")] <- SMA(l_quotes[[ch_asset]]$Volume, n = 60)
    #100 - dniowa ?rednia ruchoma dla wolumenu
    #df_attributes[paste("MA100_wolumen_", ch_asset, sep = "")] <- SMA(l_quotes[[ch_asset]]$Volume, n = 100)
    #____________________________________________________________________________________________________________

    #minMax <- MinMax(l_quotes[[ch_asset]]$High,l_quotes[[ch_asset]]$Low)
    #R??nica miedzy max a min kursem danego dnia (w %)
    #df_attributes[paste("minmax", ch_asset, sep = "")] <- minMax
    #3-dniowa ?rednia ruchoma dla atrybutu ?minmax?
    #df_attributes[paste("MA3_minmax_", ch_asset, sep = "")] <- SMA(minMax, n = 3)
    #15-dniowa ?rednia ruchoma dla atrybutu ?minmax?
    #df_attributes[paste("MA15_minmax_", ch_asset, sep = "")] <- SMA(minMax, n = 15)
    #60-dniowa ?rednia ruchoma dla atrybutu ?minmax?
    #df_attributes[paste("MA60_minmax_", ch_asset, sep = "")] <- SMA(minMax, n = 60)
    #____________________________________________________________________________________________________________

    #Stochastic 8 days indicator
    #Stochastic14 <- stoch(l_quotes[[ch_asset]], nFastK=14, nFastD=14, nSlowD=7, bounded=TRUE, smooth=1)
    #df_attributes[paste("StochFastK_", ch_asset, sep = "")] <- Stochastic[1:nrow(df_attributes),1]
    #df_attributes[paste("StochFastD_", ch_asset, sep = "")] <- Stochastic[1:nrow(df_attributes),2]
    #df_attributes[paste("StochSlowD_", ch_asset, sep = "")] <- Stochastic[1:nrow(df_attributes),3]
    #df_attributes[paste("StochOsc14_", ch_asset, sep = "")] <- Stochastic14[1:nrow(df_attributes),2] - Stochastic14[1:nrow(df_attributes),3]
    # Stochastic 100 days
    #Stochastic100 <- stoch(HLC, nFastK=100, nFastD=100, nSlowD=25, bounded=TRUE, smooth=1)
    #df_attributes[paste("StochOsc100_", ch_asset, sep = "")] <- Stochastic100[1:nrow(df_attributes),2] - Stochastic100[1:nrow(df_attributes),3]

    #takie niewiadomoco
    #df_attributes[paste("Stochastic60_", ch_asset, sep ="")] <- EMA(stoch(HLC, nFastK=14, nFastD=3, nSlowD=3, bounded=TRUE, smooth=1), n=60, wilder = TRUE)
    #____________________________________________________________________________________________________________

    #Brak informacji o transakcjach...
    #Liczba transakcji na danym walorze w danej sesji
    #df_attributes[paste("Transakcje_", ch_asset, sep = "")] <- l_quotes[[ch_asset]]$Volume
    #5-dniowa ?rednia ruchoma dla atrybutu ?Transakcje?
    #df_attributes[paste("MA5_Transakcje_", ch_asset, sep = "")] <- SMA(l_quotes[[ch_asset]]$Close, n = 5)
    #30-dniowa ?rednia ruchoma dla atrybutu ?Transakcje?
    #df_attributes[paste("MA30_Transakcje_", ch_asset, sep = "")] <- SMA(l_quotes[[ch_asset]]$Close, n = 30)
    #Miernik ?redniej warto?ci transakcji (Obr?t/Transakcje)
    #df_attributes[paste("W_Trans_", ch_asset, sep = "")] <-
    #____________________________________________________________________________________________________________
    # "If the price is making new highs, and the CCI is not, then a price correction is likely." dlatego taka ponizszy atrybut

    # df_attributes[paste("CCIind14_", ch_asset, sep = "")] <-  EMA(ReturnOnInvestment(l_quotes[[ch_asset]], 8), 16) / CCI(l_quotes[[ch_asset]], n = 8, maType = "EMA", c=0.015)  # wartosci posrednie to sygnal trendu wzrostowego, wartosci skrajne to zmiana trendu
    #df_attributes[paste("CCIind100_", ch_asset, sep = "")] <-  ReturnOnInvestment(l_quotes[[ch_asset]], 100) / CCI(l_quotes[[ch_asset]], n = 100, maType = "EMA", c=0.015)

    #df_attributes[paste("CCI15_", ch_asset, sep = "")] <- CCI(HLC, n = 15)
    #df_attributes[paste("CCI60_", ch_asset, sep = "")] <- CCI(HLC, n = 60)
    #df_attributes[paste("MA30_CCI_", ch_asset, sep = "")] <- SMA(CCI(l_quotes[[ch_asset]]$Close), n = 30)
    #____________________________________________________________________________________________________________

    # df_attributes[paste("RSI8_", ch_asset, sep = "")] <-  abs(ReturnOnInvestment(l_quotes[[ch_asset]], 8) - 1) + (RSI(l_quotes[[ch_asset]], n=8) / 100) # mo?e dzia?a? lepiej ni? powy?ej, ale i tak problem z rozr??nieniem co jest dok?adnie spadkiem a co wzrostem wg reguly technicznej...
    #df_attributes[paste("RSI100_", ch_asset, sep = "")] <- abs(ReturnOnInvestment(l_quotes[[ch_asset]], 100) - 1) + (RSI(l_quotes[[ch_asset]], n=100) / 100)
    #df_attributes[paste("RSI30updn_", ch_asset, sep = "")] <- RSI(l_quotes[[ch_asset]]$Close, n=30, maType=list(maUp=list(EMA,ratio=1/5),maDown=list(WMA,wts=1:10)))
    #df_attributes[paste("MA30_RSI_", ch_asset, sep = "")] <- SMA(RSI(l_quotes[[ch_asset]]$Close, n=14), n = 30)
    #____________________________________________________________________________________________________________

    #df_attributes[paste("ROC_", ch_asset, sep = "")] <-
    #df_attributes[paste("Momentum_", ch_asset, sep = "")] <-

    #if(nrow(l_quotes[["wig"]]))
    # df_attributes[paste("beta", ch_asset, sep = "")] <- Beta(l_quotes[[ch_asset]]$Close, l_quotes[["wig"]]$Close)
  }

  return(df_attributes)
}

#endregion
# -----------------------------------------------------------------
#region delayed Attributes

# CorrectAttributes <- function(df_attributes, l_intersectQuotes, i_delay, lo_naOmit)
# {
#   i_delayPeriod <- nrow(df_attributes) - i_delay
#
#   df_attributes[(i_delay + 1) : nrow(df_attributes), ] <- df_attributes[1 : i_delayPeriod, ]
#   df_attributes[1 : i_delay, ] <- NA
#
#   if(lo_naOmit)
#     df_attributes <- na.omit(df_attributes)
#
#   return(df_attributes)
# }
