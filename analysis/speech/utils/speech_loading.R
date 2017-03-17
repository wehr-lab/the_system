
load_generalization <- function(spfiles=spfiles,keep_cols=keep_cols,minsesh=FALSE,tok_remap=FALSE){
  gendat <- data.frame(setNames(replicate(length(keep_cols),numeric(0),simplify = F),keep_cols))
  prog_bar <- txtProgressBar(min=0,max=length(spfiles),width=20,initial=0,style=3)
  i=0
  # Loop through files
  for (f in spfiles){
    i=i+1
    setTxtProgressBar(prog_bar,i)
    
    sp <- read.csv(f)
    sp.gens <- subset(sp,(step==15) & (!is.na(correct) & (!is.na(response))),select=keep_cols)
    
    # If we only want the last n sessions, give minsesh as an int. Defaults to False. 
    if (minsesh){
      this_minsesh <- max(unique(sp.gens$session))-5
      sp.gens <- subset(sp.gens,session>=this_minsesh,select=keep_cols)
    } 
    
    # Fix speaker # for mice that use the new stimmap
    # If we want to remap tokens so all ID's are the same (ie. for token analysis), set TRUE
    # If we want to keep token maps so that learning conditions are the same (ie. for learning analysis), set FALSE
    mname <- sp.gens[1,]$mouse
    if (tok_remap == TRUE){
      if ((mname == "7007")|(mname == "7012")|(mname == "7058")|(mname == "7118")|(mname == "7120")){
        sp.gens.temp <- sp.gens #Make a copy so we don't run into recursive changes
        sp.gens.temp[sp.gens$speaker == 1,]$speaker <- 5
        sp.gens.temp[sp.gens$speaker == 2,]$speaker <- 4
        sp.gens.temp[sp.gens$speaker == 3,]$speaker <- 1
        sp.gens.temp[sp.gens$speaker == 4,]$speaker <- 2
        sp.gens.temp[sp.gens$speaker == 5,]$speaker <- 3
        sp.gens <- sp.gens.temp
      }
    }
    
    gendat <- rbind(gendat,sp.gens)
  }
  return(gendat)
}


load_timeseries <- function(spfiles=spfiles,winsize=winsize,ci=FALSE,minsesh=FALSE){
  # If confidence intervals were requested, make space in the dataframe
  if(ci){
    stepdat <- data.frame(setNames(replicate(11,numeric(0),simplify = F),c("mouse","rm","cilo","cihi","trialnum","date","session","step","level","diff","ntr")))
  } else {
    stepdat <- data.frame(setNames(replicate(9,numeric(0),simplify = F),c("mouse","rm","trialnum","date","session","step","level","diff","ntr")))
  }
  
  prog_bar <- txtProgressBar(min=0,max=length(spfiles),width=20,initial=0,style=3)
  i=0
  
  # Roll through files, compute rolling average of corrects & confidence intervals
  for (f in spfiles){
    
    fin <- read.csv(f)
    mname <- fin[1,]$mouse
    
    #Filter Data
    fin <- fin[(fin$step<=15 & fin$step>=5),]
    fin <- fin[!is.na(fin$correct),]
    
    #Rollmean & rollCI
    fin.z <- zoo(fin$correct,seq.int(1,nrow(fin)))
    fin.z.rm <- rollmean(fin.z,winsize,align="left")
    if(ci){
      fin.z.ci <- rollapply(fin.z,winsize,FUN = function(zz) binom.confint(sum(zz),winsize,conf.level=0.95,method="exact")[5:6],align="left")
    }
    
    #Make temporary dataframe and append
    if(ci){
      temp.df <- data.frame(setNames(replicate(11,numeric(0),simplify = F),c("mouse","rm","cilo","cihi","trialnum","date","session","step","level","diff","ntr")))
    } else {
      temp.df <- data.frame(setNames(replicate(9,numeric(0),simplify = F),c("mouse","rm","trialnum","date","session","step","level","diff","ntr")))
    }
    
    temp.df[1:length(fin.z.rm),]$rm <- fin.z.rm
    if(ci){
      temp.df[1:length(fin.z.ci),]$cilo <- fin.z.ci$lower
      temp.df[1:length(fin.z.ci),]$cihi <- fin.z.ci$upper
    }
    temp.df[1:nrow(temp.df),]$date <- fin[winsize:(length(fin.z.rm)-1+winsize),]$date
    temp.df[1:nrow(temp.df),]$session <- fin[winsize:(length(fin.z.rm)-1+winsize),]$session
    temp.df[1:nrow(temp.df),]$step <- fin[winsize:(length(fin.z.rm)-1+winsize),]$step
    temp.df[1:nrow(temp.df),]$trialnum <- seq(1,nrow(temp.df))
    temp.df[1:nrow(temp.df),]$diff <- cumsum(c(0,as.numeric(diff(temp.df$step))!=0))
    temp.df$mouse <- mname
    temp.df$ntr <- nrow(temp.df)
    
    #Set Levels - steps 5,6 are same level, etc.
    temp.df[(temp.df$step==5|temp.df$step==6),]$level <- 1
    temp.df[(temp.df$step==7|temp.df$step==8),]$level <- 2
    temp.df[(temp.df$step==9|temp.df$step==10),]$level <- 3
    temp.df[(temp.df$step==11|temp.df$step==12),]$level <- 4
    temp.df[(temp.df$step==13),]$level <- 5
    temp.df[(temp.df$step==15),]$level <- 6
    
    
    stepdat <- rbind(stepdat,temp.df)
    
    i=i+1
    setTxtProgressBar(prog_bar,i)
  }
  return(stepdat)
}
