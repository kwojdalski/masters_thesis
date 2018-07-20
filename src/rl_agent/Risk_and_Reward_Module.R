
require("evir")
require("spd")
require("fExtremes")
require('fGarch') 
require(magrittr)
require(ggplot2)
library(pryr)


(1+cumsum(rnorm(1000)/100))%>%data.frame(x=.)%>%ggplot(aes(y=x,x=seq(1,to=length(x))))+geom_line()
observations<-data.ES$x
position<-1
tailFraction<-0.05
eval(Estimate_Agent_Risk(observations=observations, position=position, tailFraction=tailFraction))
Estimate_Agent_Risk= function(observations, position, tailFraction){
 #inputs
  # observations nedd to be difference of log(price), one vector, observations from past window. at least 100, but the more the merier( it increases coputational time )
  # if you have a pair of assets then give this function the price of the one in terms of another
  # position is short(-1) or long(1)
  # tailFraction is the quantile we are estimating the risk measures on. Standard choice is 0.05
  DataSize=length(observations);
  scale = sd(observations);
  

  
  # fitting Aparch model, could be Garch model for the program to work faster. For APARCH see http://vlab.stern.nyu.edu/doc/5?topic=mdls
  garchFit1 = garchFit(formula=~aparch(1, 1), observations,trace=FALSE,cond.dist="sstd")


res = garchFit1@residuals / garchFit1@sigma.t


#Fitting semi parametrical distribution with GPD tails: package "spd"
# probably one could use a diffrent function to fit tails(f.e gpd()), but we needed whole distribution for later simulations
kernelGPDfit= spdfit(res, upper = 1-tailFraction, lower = tailFraction, tailfit="GPD", type = c("mle"), kernelfit = c("normal"), information = c("observed"))

if(position==1){
 a<- kernelGPDfit@fit$lowerFit
 b=tailFraction
}else{
  a<- kernelGPDfit@fit$upperFit
  b=1-tailFraction
}

# Normally you would have a simulation to get the values of risk measure for forward values.(sum of consecutive realizations)
# but as we 
VAR=qspd(b,kernelGPDfit,TRUE)
ES=gpdMoments(xi=a@fit$par.ests[1], mu = 0, beta = a@fit$par.ests[1])$mean

if(position==1){
  ES=VAR-ES
}else(
  ES=VAR+ES)



#rescalling
VAR= VAR*garchFit1@sigma.t[DataSize]
ES= ES*garchFit1@sigma.t[DataSize]
sd=garchFit1@sigma.t[DataSize]

#outputs
# OveralScale is  an sd(data)
# sd is filtered sd out of GARCh like model
#VAr is a risk measure. It is a value of the given vale between 0 and 1
#ES is a risk measure( conditional expectation ofloss , conditioned on the event that loss is below quantile E(r|r<quantile)
out=list(overalScale=scale,sd=sd,VAR=VAR,ES=ES)
}



PerformancePairTrading<-function(position,pair,coef,spread,cost=0.001,continuous=TRUE){

  n <- NROW(spread)

  #for applying transaction costs
  diff_pos <- c(0,diff(position)) # returns 1 or -1 when a position is entered
  
  gross_pnl <- numeric(NROW(pair[,1])) + 1 
  gross_pnl[is.na(gross_pnl)]<-0  
  
  if(continuous==TRUE){
    gross_pnl<-position[1:NROW(pair[,1])]*(diff.xts(log(pair[,1]))-coef[2]*diff.xts(log(pair[,2])))
    gross_pnl[is.na(gross_pnl)]<-0
    gross_pnl_cum<-cumsum(gross_pnl)
  } else{
    gross_pnl<-position[1:NROW(pair[,1])] *(diff.xts(pair[,1])/pair[,1]-coef[2]*diff.xts(pair[,2])/pair[,2])+1
    gross_pnl[is.na(gross_pnl)]<-0  
    gross_pnl_cum<-cumproduct(gross_pnl)-1
    
  }
  
  
  ntransTemp<- c(0,abs(diff(position)))*cost
  ntransTempCum<-cumsum(ntransTemp)
  
  net_pnl <- gross_pnl - ntransTemp[1:NROW(pair[,1])]
    if (continuous==TRUE){
      net_pnlCum<-cumsum(net_pnl)
    } else{
      net_pnlCum<-cumprod(1+net_pnl)
    }

NetSharpe<-sharpe(net_pnlCum,scale = sqrt(252 * 6.5 * 60))
GrossSharpe<-sharpe(gross_pnl_cum,scale = sqrt(252 * 6.5 * 60))

print(cat("\nThe Net Profit is: ",tail(net_pnlCum,1),"\n ... and the Sharpe Ratio for this dataset is: ",round(NetSharpe,digits=6),"\n"))
output<-list(ntrans=ntransTemp,FinalPnlNet=tail(net_pnlCum,1),TotalCost=tail(ntransTempCum,1),SharpeRatio=NetSharpe,NetPnl=net_pnl,
             GrossPnl=tail(gross_pnl_cum,1),GrossSharpeRatio=GrossSharpe)
}








Performance<-function(x,underlying=data.ES2$ES,cost=4,index.point=50){
  TempGross <- ifelse(is.na(x * diff.xts(underlying)),
                      0,
                      x * diff.xts(underlying)*index.point
  )
  
  TempCumSum<-cumsum(coredata(TempGross))
  
  ##### Calculation of transaction cost
  ntransTemp<-c(); 
  ntransTemp<- abs(diff.xts(x))
  ntransTemp[1] <- 0
  ntransTemp<-as.xts(ntransTemp,index(x))
  agg.ntrans<-aggregate(ntransTemp,as.Date(index(x)),FUN=sum)
  ntransTemp2<-cumsum(ntransTemp)
  ######## Net Profit of a single transaction
  PnlNet <- TempGross - ntransTemp * cost
  
  ######## Cumulated gross and net profit
  PnlNet2<-cumsum(PnlNet)
  Gross2<-cumsum(TempGross)
  
  
  ################## Net sharpe ratio #####
  
  IndSharpe<-sharpe(coredata(PnlNet2),scale = sqrt(252 * 6.5 * 60))
  GrossSharpe<-sharpe(coredata(Gross2),scale = sqrt(252 * 6.5 * 60))
  
  
  
  print(cat("\nThe average daily number of transaction is: ", mean(agg.ntrans),
            "\nThe Net Profit is: ",tail(PnlNet2,1),"\n ... and the Sharpe Ratio for this dataset is: ",round(IndSharpe,digits=6),"\n"))
  output<-list(ntrans=ntransTemp,agg.ntrans=agg.ntrans,FinalPnlNet=tail(PnlNet2,1),NumberOfTransactions=tail(ntransTemp2,1),SharpeRatio=IndSharpe,PnlNet=PnlNet2,GrossPnl=tail(Gross2,1),GrossSharpeRatio=GrossSharpe)
}

