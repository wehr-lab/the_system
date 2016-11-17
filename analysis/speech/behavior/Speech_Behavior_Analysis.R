################################################
# Collected functions for speech data analysis 
# -JLS 10.5.16
#
# Structure:
#   -File & parameter specifications
#   -Data manipulation functions
#   -Plotting functions
##################################################

## Library Imports
require(zoo)
require(ggplot2)
require(binom)
require(plyr)
require(dplyr)
require(scales)
require(reshape)
require(signal)
require(utils)
require(RColorBrewer)
require(colorspace)

################################################
## List Generalizers & Date of first gen.
data_dir <- "~/Documents/speechData/"
data_files <- list.files(data_dir,"*.csv")
for(f in data_files){
  sp <- read.csv(paste(data_dir,f,sep=""))
  if(any(sp$step==15)){
    print(f)
    datenum <- sp[which(sp$step==15),]$date[1]
    print(as.Date(datenum,origin="0000-01-01")-1)
  }
}

## List Max Step
steplist <- vector()
for(f in data_files){
  sp <- read.csv(paste(data_dir,f,sep=""))
  print(f)
  print(max(sp$step))
  steplist <- append(steplist,as.numeric(max(sp$step)),after=length(steplist))
}
table(steplist)


## Speech files
spfiles <- c("~/Documents/speechData/6924.csv",
             "~/Documents/speechData/6925.csv",
             "~/Documents/speechData/6926.csv",
             "~/Documents/speechData/6927.csv",
             "~/Documents/speechData/6928.csv",
             "~/Documents/speechData/6960.csv",
             "~/Documents/speechData/6964.csv",
             "~/Documents/speechData/6965.csv",
             "~/Documents/speechData/6966.csv",
             "~/Documents/speechData/6967.csv",
             "~/Documents/speechData/7007.csv",
             "~/Documents/speechData/7012.csv",
             "~/Documents/speechData/7058.csv")

#spfiles <- c("~/Documents/speechData/7012.csv")

# Values we want to keep from the dataframe
keep_cols <- c("consonant","speaker","vowel","token","correct","gentype","step","session","date","response")

# Size of window to use in timeseries
winsize <- 300

# Location to save files
basedir <- "/Users/Jonny/Dropbox/Lab Self/inFormants/Analysis/Plots/"

################################################
## Data Loading & cleaning

# Generalization Data Only
# Loop through files, grab columns that we want, clean data as described within function

load_generalization <- function(spfiles=spfiles,keep_cols=keep_cols,minsesh=FALSE){
  gendat <- data.frame(setNames(replicate(11,numeric(0),simplify = F),append(keep_cols,"mouse")))
  prog_bar <- txtProgressBar(min=0,max=length(spfiles),width=20,initial=0,style=3)
  i=0
  # Loop through files
  for (f in spfiles){
    i=i+1
    setTxtProgressBar(prog_bar,i)
    
    sp <- read.csv(f)
    sp.gens <- subset(sp,(step==15) & (!is.na(correct)),select=keep_cols)
    
    # If we only want the last n sessions, give minsesh as an int. Defaults to False. 
    if (minsesh){
      this_minsesh <- max(unique(sp.gens$session))-5
      sp.gens <- subset(sp.gens,session>=this_minsesh,select=keep_cols)
    } 

    # Get mouse name & add to dataframe
    mname <- substr(f,24,27)
    sp.gens$mouse <- mname
    
    # Fix speaker # for mice that use the new stimmap
    if ((mname == "7007")|(mname == "7012")){
      sp.gens.temp <- sp.gens #Make a copy so we don't run into recursive changes
      sp.gens.temp[sp.gens$speaker == 1,]$speaker <- 5
      sp.gens.temp[sp.gens$speaker == 2,]$speaker <- 4
      sp.gens.temp[sp.gens$speaker == 3,]$speaker <- 1
      sp.gens.temp[sp.gens$speaker == 4,]$speaker <- 2
      sp.gens.temp[sp.gens$speaker == 5,]$speaker <- 3
      sp.gens <- sp.gens.temp
    }
    
    #### Clean data from excess tokens - refer to "Phonumber Switches.txt"
    ### First delete pure excess
    ## Delete >3 tokens
    # Ira
    sp.gens <- sp.gens[!(sp.gens$speaker==2 & sp.gens$consonant==1 & sp.gens$vowel==1 & (sp.gens$token==4|sp.gens$token==5|sp.gens$token==7|sp.gens$token==8)),]
    sp.gens <- sp.gens[!(sp.gens$speaker==2 & sp.gens$consonant==2 & sp.gens$vowel==1 & sp.gens$token==4),]
    sp.gens <- sp.gens[!(sp.gens$speaker==2 & sp.gens$consonant==1 & sp.gens$vowel==2 & (sp.gens$token==6|sp.gens$token==7|sp.gens$token==8)),]
    sp.gens <- sp.gens[!(sp.gens$speaker==2 & sp.gens$consonant==2 & sp.gens$vowel==2 & sp.gens$token==5),]
    sp.gens <- sp.gens[!(sp.gens$speaker==2 & sp.gens$consonant==1 & sp.gens$vowel==3 & (sp.gens$token==4|sp.gens$token==5|sp.gens$token==6|sp.gens$token==7|sp.gens$token==8|sp.gens$token==9)),]
    sp.gens <- sp.gens[!(sp.gens$speaker==2 & sp.gens$consonant==2 & sp.gens$vowel==3 & (sp.gens$token==4|sp.gens$token==5)),]
    sp.gens <- sp.gens[!(sp.gens$speaker==2 & sp.gens$consonant==1 & sp.gens$vowel==4 & (sp.gens$token==5|sp.gens$token==6|sp.gens$token==7|sp.gens$token==8)),]
    sp.gens <- sp.gens[!(sp.gens$speaker==2 & sp.gens$consonant==2 & sp.gens$vowel==4 & (sp.gens$token==4|sp.gens$token==5|sp.gens$token==6)),]
    sp.gens <- sp.gens[!(sp.gens$speaker==2 & sp.gens$consonant==1 & sp.gens$vowel==5 & (sp.gens$token==4|sp.gens$token==5|sp.gens$token==7|sp.gens$token==8)),]
    sp.gens <- sp.gens[!(sp.gens$speaker==2 & sp.gens$consonant==2 & sp.gens$vowel==5 & sp.gens$token==4),]
    sp.gens <- sp.gens[!(sp.gens$speaker==2 & sp.gens$consonant==1 & sp.gens$vowel==6 & (sp.gens$token==6|sp.gens$token==7)),]
    sp.gens <- sp.gens[!(sp.gens$speaker==2 & sp.gens$consonant==2 & sp.gens$vowel==6 & sp.gens$token==5),]
    # Anna
    sp.gens <- sp.gens[!(sp.gens$speaker==3 & sp.gens$consonant==1 & sp.gens$vowel==1 & sp.gens$token==4),]
    sp.gens <- sp.gens[!(sp.gens$speaker==3 & sp.gens$consonant==1 & sp.gens$vowel==4 & (sp.gens$token==3|sp.gens$token==4)),] # No 3rd /gae/
    sp.gens <- sp.gens[!(sp.gens$speaker==3 & sp.gens$consonant==1 & sp.gens$vowel==6 & sp.gens$token==4),]
    
    ## Delete discarded/replaced <3 tokens
    # Ira 
    sp.gens <- sp.gens[!(sp.gens$date<736589 & sp.gens$speaker==2 & sp.gens$consonant==1 & sp.gens$vowel==1 & sp.gens$token==2),]
    sp.gens <- sp.gens[!(sp.gens$date<736589 & sp.gens$speaker==2 & sp.gens$consonant==1 & sp.gens$vowel==2 & (sp.gens$token==1|sp.gens$token==3)),]
    sp.gens <- sp.gens[!(sp.gens$date<736589 & sp.gens$speaker==2 & sp.gens$consonant==2 & sp.gens$vowel==2 & sp.gens$token==2),]
    sp.gens <- sp.gens[!(sp.gens$date<736589 & sp.gens$speaker==2 & sp.gens$consonant==1 & sp.gens$vowel==3 & sp.gens$token==3),]
    sp.gens <- sp.gens[!(sp.gens$date<736589 & sp.gens$speaker==2 & sp.gens$consonant==1 & sp.gens$vowel==4 & sp.gens$token==2),]
    sp.gens <- sp.gens[!(sp.gens$date<736589 & sp.gens$speaker==2 & sp.gens$consonant==1 & sp.gens$vowel==5 & sp.gens$token==3),]
    sp.gens <- sp.gens[!(sp.gens$date<736589 & sp.gens$speaker==2 & sp.gens$consonant==1 & sp.gens$vowel==6 & (sp.gens$token==1|sp.gens$token==2)),]
    sp.gens <- sp.gens[!(sp.gens$date<736589 & sp.gens$speaker==2 & sp.gens$consonant==2 & sp.gens$vowel==6 & sp.gens$token==1),]
    # Anna
    sp.gens <- sp.gens[!(sp.gens$date<736589 & sp.gens$speaker==3 & sp.gens$consonant==1 & sp.gens$vowel==3 & sp.gens$token==1),]
    # Dani
    sp.gens <- sp.gens[!(sp.gens$date<736589 & sp.gens$speaker==4 & sp.gens$consonant==1 & sp.gens$vowel==4 & sp.gens$token==1),]
    # Theresa
    sp.gens <- sp.gens[!(sp.gens$date<736589 & sp.gens$speaker==5 & sp.gens$consonant==1 & sp.gens$vowel==1 & sp.gens$token==2),]
    sp.gens <- sp.gens[!(sp.gens$date<736589 & sp.gens$speaker==5 & sp.gens$consonant==1 & sp.gens$vowel==4 & sp.gens$token==2),]
    sp.gens <- sp.gens[!(sp.gens$date<736589 & sp.gens$speaker==5 & sp.gens$consonant==1 & sp.gens$vowel==6 & sp.gens$token==2),]
    
    ## Renumber tokens that turned out to be the same token
    try(sp.gens[(sp.gens$speaker==4 & sp.gens$consonant==1 & (sp.gens$vowel==1|sp.gens$vowel==2|sp.gens$vowel==3|sp.gens$vowel==6) & sp.gens$token==4),]$token <- 3,silent = T)
    try(sp.gens[(sp.gens$speaker==4 & sp.gens$consonant==1 & sp.gens$vowel==4 & sp.gens$token==5),]$token <- 4,silent = T)
    try(sp.gens[(sp.gens$speaker==4 & sp.gens$consonant==2 & (sp.gens$vowel==1|sp.gens$vowel==5) & sp.gens$token==4),]$token <- 3,silent = T)
    
    ## Then realign data that had its number changed
    # Ira
    try(sp.gens[(sp.gens$date<736589 & sp.gens$speaker==2 & sp.gens$consonant==1 & sp.gens$vowel==1 & sp.gens$token==6),]$token <- 2,silent = T)
    try(sp.gens[(sp.gens$date<736589 & sp.gens$speaker==2 & sp.gens$consonant==1 & sp.gens$vowel==2 & sp.gens$token==4),]$token <- 1,silent = T)
    try(sp.gens[(sp.gens$date<736589 & sp.gens$speaker==2 & sp.gens$consonant==1 & sp.gens$vowel==2 & sp.gens$token==5),]$token <- 3,silent = T)
    try(sp.gens[(sp.gens$date<736589 & sp.gens$speaker==2 & sp.gens$consonant==2 & sp.gens$vowel==2 & sp.gens$token==4),]$token <- 2,silent = T)
    try(sp.gens[(sp.gens$date<736589 & sp.gens$speaker==2 & sp.gens$consonant==1 & sp.gens$vowel==3 & sp.gens$token==10),]$token <- 3,silent = T)
    try(sp.gens[(sp.gens$date<736589 & sp.gens$speaker==2 & sp.gens$consonant==1 & sp.gens$vowel==4 & sp.gens$token==4),]$token <- 2,silent = T)
    try(sp.gens[(sp.gens$date<736589 & sp.gens$speaker==2 & sp.gens$consonant==1 & sp.gens$vowel==5 & sp.gens$token==6),]$token <- 3,silent = T)
    try(sp.gens[(sp.gens$date<736589 & sp.gens$speaker==2 & sp.gens$consonant==1 & sp.gens$vowel==6 & sp.gens$token==5),]$token <- 1,silent = T)
    try(sp.gens[(sp.gens$date<736589 & sp.gens$speaker==2 & sp.gens$consonant==1 & sp.gens$vowel==6 & sp.gens$token==4),]$token <- 2,silent = T)
    try(sp.gens[(sp.gens$date<736589 & sp.gens$speaker==2 & sp.gens$consonant==2 & sp.gens$vowel==6 & sp.gens$token==4),]$token <- 1,silent = T)
    # Anna
    try(sp.gens[(sp.gens$date<736589 & sp.gens$speaker==3 & sp.gens$consonant==1 & sp.gens$vowel==3 & sp.gens$token==4),]$token <- 1,silent = T)
    # Dani
    try(sp.gens[(sp.gens$date<736589 & sp.gens$speaker==4 & sp.gens$consonant==1 & sp.gens$vowel==4 & sp.gens$token==4),]$token <- 1,silent = T)
    # Theresa
    try(sp.gens[(sp.gens$date<736589 & sp.gens$speaker==5 & sp.gens$consonant==1 & sp.gens$vowel==1 & sp.gens$token==4),]$token <- 2,silent = T)
    try(sp.gens[(sp.gens$date<736589 & sp.gens$speaker==5 & sp.gens$consonant==1 & sp.gens$vowel==4 & sp.gens$token==4),]$token <- 2,silent = T)
    try(sp.gens[(sp.gens$date<736589 & sp.gens$speaker==5 & sp.gens$consonant==1 & sp.gens$vowel==6 & sp.gens$token==4),]$token <- 2,silent = T)
    
    # Testing...
    if ((mname == "7007")|(mname == "7012")){
      sp.gens.temp <- sp.gens
      sp.gens.temp[sp.gens$speaker == 5,]$speaker <- 1
      sp.gens.temp[sp.gens$speaker == 4,]$speaker <- 2
      sp.gens.temp[sp.gens$speaker == 1,]$speaker <- 3
      sp.gens.temp[sp.gens$speaker == 2,]$speaker <- 4
      sp.gens.temp[sp.gens$speaker == 3,]$speaker <- 5
      sp.gens <- sp.gens.temp
    }
    
    #prevent overlapping gentypes from before sampling was not mutually exclusive.
    # First knock everything not a 3 to a 2
    try(sp.gens[sp.gens$date<736589 & (sp.gens$vowel==3|sp.gens$vowel==2|sp.gens$vowel==1),]$gentype <- 2,silent = T) 
    # Then specifically define all the 1's.
    try(sp.gens[sp.gens$date<736589 & (sp.gens$speaker==1|sp.gens$speaker==2) & (sp.gens$vowel==1|sp.gens$vowel==2) & (sp.gens$token==1|sp.gens$token==2),]$gentype <- 1,silent = T)
    try(sp.gens[sp.gens$date<736589 & (sp.gens$speaker==1|sp.gens$speaker==2) & sp.gens$vowel==3 & sp.gens$token==1,]$gentype <- 1,silent = T)
    
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
    mname <- substr(f,24,27)
    
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

#################################################
## Data Reshaping & Summarization

# Summarize data by....
gendat.type <- ddply(gendat,.(gentype),summarize, meancx = mean(correct),cilo = binom.confint(sum(correct),length(correct),conf.level=0.95,method="exact")[[5]],cihi = binom.confint(sum(correct),length(correct),conf.level=0.95,method="exact")[[6]],nobs = length(correct),nmice=length(unique(mouse)))
gendat.mouse.type <- ddply(gendat.mouse,.(gentype),summarize, mean_meancx = mean(meancx),cilo = (mean(meancx)-((qt(.975,df=length(unique(mouse))-1))*(sd(meancx)/sqrt(length(unique(mouse)))))),cihi = (mean(meancx)+((qt(.975,df=length(unique(mouse))-1))*(sd(meancx)/sqrt(length(unique(mouse)))))),nobs = sum(nobs),nmice=length(unique(mouse)))
gendat.mouse <- ddply(gendat,.(mouse,gentype),summarize, meancx = mean(correct),cilo = binom.confint(sum(correct),length(correct),conf.level=0.95,method="exact")[[5]],cihi = binom.confint(sum(correct),length(correct),conf.level=0.95,method="exact")[[6]],nobs = length(correct))
gendat.binom <- ddply(gendat,.(mouse,gentype),summarize, meancx = mean(correct),binom = binom.test(sum(correct),length(correct),p=0.5,conf.level=0.95,alternative="greater")[[3]])
gendat.tokenbinom <- ddply(gendat,.(mouse,vowel,speaker,consonant),summarize, meancx = mean(correct),binom = binom.test(sum(correct),length(correct),p=0.5,conf.level=0.95,alternative="greater")[[3]])

gendat.token <- ddply(gendat,.(vowel,speaker,consonant,token),summarize, meancx = mean(correct),cilo = binom.confint(sum(correct),length(correct),conf.level=0.95,method="exact")[[5]],cihi = binom.confint(sum(correct),length(correct),conf.level=0.95,method="exact")[[6]],nobs = length(correct))
gendat.tokresp <- ddply(gendat,.(vowel,speaker,consonant,token),summarize, meancx = mean(correct),cilo = binom.confint(sum(correct),length(correct),conf.level=0.95,method="exact")[[5]],cihi = binom.confint(sum(correct),length(correct),conf.level=0.95,method="exact")[[6]],nobs = length(correct))

gendat.token <- ddply(gendat,.(vowel,speaker,consonant,token),summarize, meancx = mean(correct),cilo = binom.confint(sum(correct),length(correct),conf.level=0.95,method="exact")[[5]],cihi = binom.confint(sum(correct),length(correct),conf.level=0.95,method="exact")[[6]],nobs = length(correct))
gendat.tokmus <- ddply(gendat,.(mouse,gentype,vowel,speaker,consonant,token),summarize, meancx = mean(correct),cilo = binom.confint(sum(correct),length(correct),conf.level=0.95,method="exact")[[5]],cihi = binom.confint(sum(correct),length(correct),conf.level=0.95,method="exact")[[6]],nobs = length(correct))
gendat.vowel <- ddply(gendat,.(mouse,consonant,vowel),summarize, meancx = mean(correct),cilo = binom.confint(sum(correct),length(correct),conf.level=0.95,method="exact")[[5]],cihi = binom.confint(sum(correct),length(correct),conf.level=0.95,method="exact")[[6]],nobs = length(correct))
gendat.speaker <- ddply(gendat,.(speaker,consonant,vowel),summarize, meancx = mean(correct),cilo = binom.confint(sum(correct),length(correct),conf.level=0.95,method="exact")[[5]],cihi = binom.confint(sum(correct),length(correct),conf.level=0.95,method="exact")[[6]],nobs = length(correct))

# Adjust timeseries correct value by step
adjust_ts_step <- function(ts){
  ts.adj <- ts
  
  ts.adj[(ts.adj$step == 7 | ts.adj$step == 8),]$rm <- ((ts.adj[(ts.adj$step == 7 | ts.adj$step == 8),]$rm-.4)*1.67) + 1
  ts.adj[(ts.adj$step == 9 | ts.adj$step == 10),]$rm <- ((ts.adj[(ts.adj$step == 9 | ts.adj$step == 10),]$rm-.4)*1.67) + 2
  ts.adj[(ts.adj$step == 11 | ts.adj$step == 12),]$rm <- ((ts.adj[(ts.adj$step == 11 | ts.adj$step == 12),]$rm-.4)*1.67) + 3
  ts.adj[(ts.adj$step == 13),]$rm <- ((ts.adj[(ts.adj$step == 13),]$rm-.4)*1.67) + 4
  ts.adj[(ts.adj$step == 15),]$rm <- ((ts.adj[(ts.adj$step == 15),]$rm-.4)*1.67) + 5
  
  # If we have confidence intervals... 
  if("cilo" %in% names(ts.adj)){
    ts.adj[(ts.adj$step == 7 | ts.adj$step == 8),]$cilo <- ((ts.adj[(ts.adj$step == 7 | ts.adj$step == 8),]$cilo-.4)*1.67) + 1
    ts.adj[(ts.adj$step == 9 | ts.adj$step == 10),]$cilo <- ((ts.adj[(ts.adj$step == 9 | ts.adj$step == 10),]$cilo-.4)*1.67) + 2
    ts.adj[(ts.adj$step == 11 | ts.adj$step == 12),]$cilo <- ((ts.adj[(ts.adj$step == 11 | ts.adj$step == 12),]$cilo-.4)*1.67) + 3
    ts.adj[(ts.adj$step == 13),]$cilo <- ((ts.adj[(ts.adj$step == 13),]$cilo-.4)*1.67) + 4
    ts.adj[(ts.adj$step == 15),]$cilo <- ((ts.adj[(ts.adj$step == 15),]$cilo-.4)*1.67) + 5
    
    ts.adj[(ts.adj$step == 7 | ts.adj$step == 8),]$cihi <- ((ts.adj[(ts.adj$step == 7 | ts.adj$step == 8),]$cihi-.4)*1.67) + 1
    ts.adj[(ts.adj$step == 9 | ts.adj$step == 10),]$cihi <- ((ts.adj[(ts.adj$step == 9 | ts.adj$step == 10),]$cihi-.4)*1.67) + 2
    ts.adj[(ts.adj$step == 11 | ts.adj$step == 12),]$cihi <- ((ts.adj[(ts.adj$step == 11 | ts.adj$step == 12),]$cihi-.4)*1.67) + 3
    ts.adj[(ts.adj$step == 13),]$cihi <- ((ts.adj[(ts.adj$step == 13),]$cihi-.4)*1.67) + 4
    ts.adj[(ts.adj$step == 15),]$cihi <- ((ts.adj[(ts.adj$step == 15),]$cihi-.4)*1.67) + 5
  }
  return(ts.adj)
}

################################################
## Basic Stats
mdem <- read.csv("~/Dropbox/Lab Self/inFormants/Mouse_Demographics.csv")
# Set vars as factors
mdem$Status <- as.factor(mdem$Status)
mdem$Sex    <- as.factor(mdem$Sex)
mdem$Box    <- as.factor(mdem$Box)
mdem$Strain <- as.factor(mdem$Strain)
mdem$Litter <- as.factor(mdem$Litter)
# Subset
mdem.fin <- mdem[(mdem$Status==1 | mdem$Status==3),]
mdem.gen <- mdem[mdem$Status==3,]

## Logistic Regressions
mdem.lm.sex <- glm(Status ~ Sex, data=mdem.fin, family="quasibinomial")
mdem.lm.box <- glm(Status ~ Box, data=mdem.fin, family="quasibinomial")
mdem.lm.str <- glm(Status ~ Strain, data=mdem.fin, family="quasibinomial")
mdem.lm.lit <- glm(Status ~ Litter, data=mdem.fin, family="quasibinomial")
mdem.lm.age <- glm(Status ~ AgeStart, data=mdem.fin, family="quasibinomial")

summary(mdem.lm.sex)
anova(mdem.lm.sex,test="Chisq")
summary(mdem.lm.box)
anova(mdem.lm.box,test="Chisq")
summary(mdem.lm.str)
anova(mdem.lm.str,test="Chisq")
summary(mdem.lm.lit)
anova(mdem.lm.lit,test="Chisq")
summary(mdem.lm.age)
anova(mdem.lm.age,test="Chisq")


# Plot log regression
plot_log <- function(xvar,yvar,dat){
ggplot(dat, aes_string(x=xvar,colour=yvar,fill=yvar),environment = environment()) + geom_density(alpha=0.25)  + stat_smooth(aes_string(y=yvar, colour=NULL, fill=NULL),method="glm", method.args = list(family="binomial"))
}
  #################################################
## Plotting

# Timeseries Area Plot
plot_ts_area <- function(ts){
  # Reorder mice by number of trials done
  ts$mouse <- factor(ts$mouse, levels=ts[order(ts$ntr,decreasing=TRUE),"mouse"])
  ts.dec <- ts[seq(1,nrow(ts),10),]
  disc_cols <- c("#D53E4F","#FC8D59","#FEE08B","#E6F598","#99D594","#3288BD")
  disc_cols_dark <- c("#B71A33","#DE7237","#E1C46C","#CAD97A","#7DB978","#147BB0")

  p.area <- ggplot(data=ts.dec,aes(x=trialnum,fill=as.factor(level))) +
    scale_fill_manual(values=disc_cols,guide=guide_legend(title="Level",nrow=1,title.position="top",label.position="bottom",direction="horizontal"))+
    scale_color_manual(values=disc_cols_dark,guide=guide_legend(title="Level",nrow=1,title.position="top",label.position="bottom",direction="horizontal"))+
    geom_ribbon(aes(ymax=rm,ymin=(1-rm),group=diff,colour=as.factor(level)),alpha=0.8,size=0.4)+
    scale_y_continuous(breaks=c(.25,.5,.75),labels=c("75%","50%","75%"))+
    #scale_x_continuous(name="# of Trials") +
    facet_grid(mouse~.,scales="fixed") +
    theme(panel.background = element_rect(fill=alpha('gray',0.2)),
          plot.background = element_rect(fill="transparent",colour=NA),
          panel.grid.minor = element_blank(),
          strip.text = element_text(size=rel(1)),
          axis.title.y = element_blank(),
          axis.text.y = element_text(size=rel(1)),
          #axis.title.x = element_text(size=rel(1.2),margin=margin(6,2,2,2,unit="pt")),    
          axis.title.x = element_blank(),
          axis.text.x = element_text(size=rel(2),margin=margin(6,1,1,1,unit="pt")),
          legend.position = "top",
          legend.justification = "left",
          legend.text = element_text(size=rel(1)),
          legend.title = element_text(size=rel(1.5)),
          legend.key.size = unit("0.25","cm"))
  
  ggsave(paste(basedir,"ts_area_",as.numeric(as.POSIXct(Sys.time())),".png"),plot=p.area,device="png",width=13.33,height=7.5,units="in",dpi=700,bg="transparent")
  
  return(p.area)
}

# Timeseries Level Plot
plot_ts_level <- function(ts){
  ts.dec <- ts[seq(1,nrow(ts),10),]
  stepdat <- adjust_ts_step(ts.dec)
  
  stepplot <- ggplot(stepdat,aes(x=trialnum,y=rm)) + 
    scale_x_continuous(name = "# of Trials") + 
    scale_y_continuous(breaks=c(0.16,.583,1.16,1.583,2.16,2.583,3.16,3.583,4.16,4.583,5.16,5.583),
                       minor_breaks = c(.583,1.583,2.583,3.583,4.583,5.583),
                                              labels=c("50%","75%","50%","75%","50%","75%","50%","75%","50%","75%","50%","75%"),
                                              limits=c(0,6)) + 
    geom_ribbon(aes(ymin=cilo,ymax=cihi),fill="gray",alpha=0.7,linetype=2,size=0.5) +
    geom_line(data=stepdat,aes(x=trialnum,y=rm)) +
    geom_line(aes(y=0),colour="#74828F",size=1) +
    geom_line(aes(y=1),colour="#74828F",size=1) +
    geom_line(aes(y=2),colour="#74828F",size=1) +
    geom_line(aes(y=3),colour="#74828F",size=1) +
    geom_line(aes(y=4),colour="#74828F",size=1) +
    geom_line(aes(y=5),colour="#74828F",size=1) +
    geom_line(aes(y=6),colour="#74828F",size=1) +
    geom_line(aes(y=0.16),colour="#C25B56",size=.75,linetype="longdash") + 
    geom_line(aes(y=1.16),colour="#C25B56",size=.75,linetype="longdash") + 
    geom_line(aes(y=2.16),colour="#C25B56",size=.75,linetype="longdash") + 
    geom_line(aes(y=3.16),colour="#C25B56",size=.75,linetype="longdash") + 
    geom_line(aes(y=4.16),colour="#C25B56",size=.75,linetype="longdash") + 
    geom_line(aes(y=5.16),colour="#C25B56",size=.75,linetype="longdash") + 
    theme(panel.background = element_blank(), 
          panel.grid.minor = element_blank(),
          panel.grid.major = element_line(colour=alpha("gray",0.4)),
          axis.text.x = element_text(size=rel(1.25)),
          axis.text.y = element_text(size=rel(1.25),margin=margin(t=0,r=0,b=0,l=0,unit="pt")),
          axis.title.y = element_blank(),
          axis.title.x = element_text(size=rel(1.5)),
          #axis.ticks.y = element_blank(),
          legend.position = 'none')
  stepplot
  
  ggsave(paste(basedir,"ts_step_",as.numeric(as.POSIXct(Sys.time())),".png"),plot=stepplot,device="png",width=9,height=7,units="in",dpi=700)
  
}


# Barplot split by mouse
#plot bars split by mouse
limits <- aes(ymax=cihi,ymin=cilo)
dodge <- position_dodge(width=.9)
gen.barmouse <- ggplot(gendat.mouse,aes(mouse,meancx,fill=as.factor(gentype))) + 
  geom_bar(position="dodge",stat="identity") +
  geom_errorbar(limits,position=dodge,width=0.25,size=0.4,color="white") + 
  scale_y_continuous(limits = c(.4,.95),breaks=c(.4,.5,.6,.7,.8,.9),labels=c("40%","50%","60%","70%","80%","90%"),oob = rescale_none) +
  scale_fill_brewer(palette="Set1",name="Generalization Type",labels=c("Learned","Learned Vowels, Unlearned Speakers & Tokens","Unlearned Vowels, Speakers, Tokens"))+
  #scale_fill_discrete(name="Generalization Type",labels=c("Learned","Learned Vowels, Unlearned Speakers & Tokens","Unlearned Vowels, Speakers, Tokens"))+
  xlab("Mouse ID") +
  geom_hline(yintercept = 0.5,size=1,linetype=2) +
  theme(panel.background = element_blank(), 
        legend.background = element_rect(fill="transparent",colour=NA),
        plot.background = element_rect(fill="transparent",colour=NA),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        axis.title.y = element_blank(),
        axis.text.y = element_text(size=rel(1.5),color="white"),
        axis.ticks.y = element_blank(),
        axis.text.y = element_blank(),
        axis.title.x = element_text(size=rel(1.5),color="white"),        
        axis.text.x = element_text(size=rel(1.5),color="white"),
        #legend.position = c(.4,.85),
        legend.position = "none",
        legend.text = element_text(size=rel(1.2)),
        legend.title = element_text(size=rel(1.5)))
#gen.barmouse
gendat.mouse <- data.frame(1)
gendat.mouse$mouse <- 'Classifier'
gendat.mouse$gentype <- 1
gendat.mouse$meancx <- .78
gendat.mouse <- gen.barmouse[1]

gen.barmouse <- ggplot(gendat.mouse,aes(mouse,meancx,fill=as.factor(gentype))) + 
  geom_bar(position="dodge",stat="identity") +
  #geom_errorbar(limits,position=dodge,width=0.25,size=0.4,color="white") + 
  scale_y_continuous(limits = c(.4,.95),breaks=c(.4,.5,.6,.7,.8,.9),labels=c("40%","50%","60%","70%","80%","90%"),oob = rescale_none) +
  scale_fill_brewer(palette="Set1",name="Generalization Type",labels=c("Learned","Learned Vowels, Unlearned Speakers & Tokens","Unlearned Vowels, Speakers, Tokens"))+
  #scale_fill_discrete(name="Generalization Type",labels=c("Learned","Learned Vowels, Unlearned Speakers & Tokens","Unlearned Vowels, Speakers, Tokens"))+
  #xlab("Mouse ID") +
  geom_hline(yintercept = 0.5,size=1,linetype=2) +
  theme(panel.background = element_blank(), 
        legend.background = element_rect(fill="transparent",colour=NA),
        plot.background = element_rect(fill="transparent",colour=NA),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        axis.title.y = element_blank(),
        #axis.text.y = element_text(size=rel(1.5),color="white"),
        axis.ticks.y = element_blank(),
        axis.text.y = element_blank(),
        axis.title.x = element_text(size=rel(1.5),color="white"),        
        axis.text.x = element_text(size=rel(1.5),color="white"),
        #legend.position = c(.4,.85),
        legend.position = "none",
        legend.text = element_text(size=rel(1.2)),
        legend.title = element_text(size=rel(1.5)))
#gen.barmouse

#ggsave("~/Documents/speechPlots/25_27_28_genbarmouse.svg",plot=gen.barmouse,device="svg",width=8,height=4,units="in")
ggsave(paste(basedir,"genbar_mouse_",as.numeric(as.POSIXct(Sys.time())),".png"),plot=gen.barmouse,device="png",width=9.5,height=6.5,units="in",dpi=700,bg="transparent")
ggsave(paste(basedir,"genbar_mouse_",as.numeric(as.POSIXct(Sys.time())),".png"),plot=gen.barmouse,device="png",width=1,height=6.5,units="in",dpi=700,bg="transparent")


gendat.mouse.type <- data.frame()
gendat.mouse.type
# Bar Plot across mice
limits <- aes(ymax=cihi,ymin=cilo)
dodge <- position_dodge(width=.9)
gen.bar <- ggplot(gendat.mouse.type,aes(gentype,mean_meancx,fill=as.factor(gentype))) + 
  geom_bar(position="dodge",stat="identity") +
  geom_errorbar(limits,position=dodge,width=0.25,size=0.4) + 
  scale_y_continuous(limits = c(.4,.95),breaks=c(.4,.5,.6,.7,.8,.9),labels=c("40%","50%","60%","70%","80%","90%"),oob = rescale_none) +
  #scale_fill_discrete(name="Generalization Type",labels=c("Trained","Trained Vowels, Novel Speakers & Tokens","Novel Vowels, Speakers, Tokens"))+
  scale_fill_brewer(palette="Set1")+
  xlab("Generalization Type") +
  ylab("% Correct") +
  geom_hline(yintercept = 0.5,size=1,linetype=2) +
  theme(panel.background = element_blank(), 
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        plot.background = element_rect(fill="transparent",colour=NA),
        axis.title.y = element_text(size=rel(1.5)),
        axis.text.y = element_text(size=rel(1.5)),
        #axis.title.x = element_text(size=rel(1.5)),        
        axis.text.x = element_text(size=rel(1.5)),
        axis.title.x = element_text(size=rel(1.5)),
        axis.text.x = element_text(size=rel(1.5)),
        #axis.ticks.x = element_blank(),
        #legend.position = c(.5,.85),
        legend.position = "none",
        legend.text = element_text(size=rel(.8)),
        legend.title = element_text(size=rel(1.2)))
#gen.bar

#ggsave("~/Documents/speechPlots/25_27_28_genbarmouse.svg",plot=gen.barmouse,device="svg",width=8,height=4,units="in")
ggsave(paste(basedir,"genbar_all_",as.numeric(as.POSIXct(Sys.time())),".png"),plot=gen.bar,device="png",width=3,height=6.5,units="in",dpi=700,bg="transparent")


#################
## Misc

# Colors

disc_cols <- list(color=colorRampPalette(brewer.pal(6,"Spectral"))(6))
disc_cols_dark <- list()
for(i in seq(1,length(disc_cols$color))){
  col <- as(hex2RGB(disc_cols$color[i]),"polarLUV")
  col@coords[1] <- col@coords[1]-10
  disc_cols_dark <- append(disc_cols_dark,hex(col))
}

classifications <- read.csv("~/Documents/tempnpdat/probs.csv",header = FALSE)
classifications$V1 = classifications$V1-0.5
classifications$tok <- seq(1,nrow(classifications))
classifications$type <- c(rep(1,79),rep(2,82))
classis <- ggplot(classifications, aes(x=tok,y=V1,fill=factor(type))) +
  geom_bar(stat='identity') +
  scale_y_continuous(limits = c(-.25,.25))+
  scale_fill_brewer(palette="Set1") +
  coord_flip()+ 
  theme(panel.background = element_blank(),
        legend.background = element_rect(fill="transparent",colour=NA),
        plot.background = element_rect(fill="transparent",colour=NA),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        axis.title.y = element_blank(),
        #axis.text.y = element_text(size=rel(1.5)),
        axis.ticks.y = element_blank(),
        axis.text.y = element_blank(),
        axis.title.x = element_blank(),        
        axis.text.x = element_text(size=rel(1.5)),
        #legend.position = c(.4,.85),
        legend.position = "none",
        legend.text = element_text(size=rel(1.2)),
        legend.title = element_text(size=rel(1.5)))
classis
ggsave(paste(basedir,"tokclass",as.numeric(as.POSIXct(Sys.time())),".png"),plot=classis,device="png",width=7,height=6.5,units="in",dpi=700,bg="transparent")

