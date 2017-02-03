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
require(gtable)
require(grid)
require(gridExtra)
require(cowplot)
require(ggbiplot)

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
spfiles <- c("~/Documents/fixSpeechData/6924.csv",
             "~/Documents/fixSpeechData/6925.csv",
             "~/Documents/fixSpeechData/6926.csv",
             "~/Documents/fixSpeechData/6927.csv",
             "~/Documents/fixSpeechData/6928.csv",
             "~/Documents/fixSpeechData/6960.csv",
             "~/Documents/fixSpeechData/6964.csv",
             "~/Documents/fixSpeechData/6965.csv",
             "~/Documents/fixSpeechData/6966.csv",
             "~/Documents/fixSpeechData/6967.csv",
             "~/Documents/fixSpeechData/7007.csv",
             "~/Documents/fixSpeechData/7012.csv",
             "~/Documents/fixSpeechData/7058.csv")

#spfiles <- c("~/Documents/speechData/7012.csv")

# Values we want to keep from the dataframe
keep_cols <- c("trialNumber","consonant","speaker","vowel","token","correct","gentype","step","session","date","response","target","mouse")

# Size of window to use in timeseries
winsize <- 300

# Location to save files
basedir <- "/Users/Jonny/Dropbox/Lab Self/inFormants/Analysis/Plots/"

################################################
## Data Loading & cleaning

# Generalization Data Only
# Loop through files, grab columns that we want, clean data as described within function

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
      if ((mname == "7007")|(mname == "7012")|(mname == "7058")){
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

#################################################
## Data Reshaping & Summarization

# Summarize data by....
gendat.type <- ddply(gendat,.(gentype),summarize, meancx = mean(correct),cilo = binom.confint(sum(correct),length(correct),conf.level=0.95,method="exact")[[5]],cihi = binom.confint(sum(correct),length(correct),conf.level=0.95,method="exact")[[6]],nobs = length(correct),nmice=length(unique(mouse)))
gendat.mouse.type <- ddply(gendat.mouse,.(gentype),summarize, mean_meancx = mean(meancx),meanresp=mean(response),meantarg=mean(target),cilo = (mean(meancx)-((qt(.975,df=length(unique(mouse))-1))*(sd(meancx)/sqrt(length(unique(mouse)))))),cihi = (mean(meancx)+((qt(.975,df=length(unique(mouse))-1))*(sd(meancx)/sqrt(length(unique(mouse)))))),nobs = sum(nobs),nmice=length(unique(mouse)))
gendat.mouse <- ddply(gendat,.(mouse,gentype),summarize, meancx = mean(correct),meanresp=mean(response),meantarg=mean(target),cilo = binom.confint(sum(correct),length(correct),conf.level=0.95,method="exact")[[5]],cihi = binom.confint(sum(correct),length(correct),conf.level=0.95,method="exact")[[6]],nobs = length(correct))
gendat.constype <- ddply(gendat,.(mouse,consonant,gentype),summarize, meancx = mean(correct),meanresp=mean(response),meantarg=mean(target),cilo = binom.confint(sum(correct),length(correct),conf.level=0.95,method="exact")[[5]],cihi = binom.confint(sum(correct),length(correct),conf.level=0.95,method="exact")[[6]],nobs = length(correct))

gendat.binom <- ddply(gendat,.(mouse,gentype),summarize, meancx = mean(correct),binom = binom.test(sum(correct),length(correct),p=0.5,conf.level=0.95,alternative="greater")[[3]])
gendat.tokenbinom <- ddply(gendat,.(mouse,vowel,speaker,consonant),summarize, meancx = mean(correct),binom = binom.test(sum(correct),length(correct),p=0.5,conf.level=0.95,alternative="greater")[[3]])

gendat.token <- ddply(gendat,.(consonant,speaker,vowel,token),summarize, meancx = mean(correct),cilo = binom.confint(sum(correct),length(correct),conf.level=0.95,method="exact")[[5]],cihi = binom.confint(sum(correct),length(correct),conf.level=0.95,method="exact")[[6]],nobs = length(correct))
gendat.tokresp <- ddply(gendat,.(vowel,speaker,consonant,token),summarize, meancx = mean(correct),cilo = binom.confint(sum(correct),length(correct),conf.level=0.95,method="exact")[[5]],cihi = binom.confint(sum(correct),length(correct),conf.level=0.95,method="exact")[[6]],nobs = length(correct))

#gendat.token <- ddply(gendat,.(vowel,speaker,consonant,token),summarize, meancx = mean(correct),cilo = binom.confint(sum(correct),length(correct),conf.level=0.95,method="exact")[[5]],cihi = binom.confint(sum(correct),length(correct),conf.level=0.95,method="exact")[[6]],nobs = length(correct))
gendat.tokmus <- ddply(gendat,.(mouse,consonant,speaker,vowel),summarize, meancx = mean(correct),cilo = binom.confint(sum(correct),length(correct),conf.level=0.95,method="exact")[[5]],cihi = binom.confint(sum(correct),length(correct),conf.level=0.95,method="exact")[[6]],nobs = length(correct))
gendat.tokmouse <- ddply(gendat,.(mouse,consonant,speaker,vowel,token),summarize, meancx = mean(correct))

gendat.vowel <- ddply(gendat,.(mouse,consonant,vowel),summarize, meancx = mean(correct),cilo = binom.confint(sum(correct),length(correct),conf.level=0.95,method="exact")[[5]],cihi = binom.confint(sum(correct),length(correct),conf.level=0.95,method="exact")[[6]],nobs = length(correct))
gendat.speaker <- ddply(gendat,.(consonant,speaker,vowel),summarize, meancx = mean(correct),cilo = binom.confint(sum(correct),length(correct),conf.level=0.95,method="exact")[[5]],cihi = binom.confint(sum(correct),length(correct),conf.level=0.95,method="exact")[[6]],nobs = length(correct))

gendat.bias <- ddply(gendat,.(mouse),summarize, meancx = mean(correct),meanresp = mean(response),meantarg=mean(target),nobs = length(correct))
gendat.bias$bias <- gendat.bias$meanresp-gendat.bias$meantarg
gendat.cons <- ddply(gendat,.(mouse,consonant),summarize, meancx = mean(correct),meanresp = mean(response),meantarg=mean(target),nobs = length(correct))
gendat.mouse$bias <- gendat.mouse$meanresp-gendat.mouse$meantarg
gendat.bias["conscx"] <- NA
gendat.bias["gencx"] <- NA
gendat.bias["genbias"] <- NA
gendat.bias["genconscx"] <- NA
for(m in unique(gendat.cons$mouse)){
  gendat.bias[gendat.bias$mouse == m,]$conscx <- gendat.cons[(gendat.cons$mouse == m & gendat.cons$consonant == 1),]$meancx - gendat.cons[(gendat.cons$mouse == m & gendat.cons$consonant == 2),]$meancx
  gendat.bias[gendat.bias$mouse == m,]$gencx <- gendat.mouse[(gendat.mouse$mouse == m & gendat.mouse$gentype == 3),]$meancx
  gendat.bias[gendat.bias$mouse == m,]$genbias <- gendat.mouse[(gendat.mouse$mouse == m & gendat.mouse$gentype == 3),]$bias
  gendat.bias[gendat.bias$mouse == m,]$genconscx <- gendat.constype[(gendat.constype$mouse == m & gendat.constype$consonant == 1 & gendat.constype$gentype == 3),]$meancx - gendat.constype[(gendat.constype$mouse == m & gendat.constype$consonant == 2 & gendat.constype$gentype == 3),]$meancx
}
gendat.bias$cxdiff <- gendat.bias$gencx - gendat.bias$meancx
gendat.bias$absbias <- abs(gendat.bias$bias)
gendat.bias$absgenbias <- abs(gendat.bias$genbias)

# Additional ID columns if needed (eg. for heatmap, tokenplot)
gendat.token$ID <- as.factor(paste(gendat.token$consonant,gendat.token$vowel,gendat.token$speaker,gendat.token$token))
gendat.speaker$ID <- as.factor(paste(gendat.speaker$consonant,gendat.speaker$speaker,gendat.speaker$vowel))

gendat.tokmus$ID <- as.factor(paste(gendat.tokmus$speaker,gendat.tokmus$vowel))

# Reshape gendat.mouse for regression plot
gendat.mouse.rs <- cast(gendat.mouse,mouse~gentype,value="meancx")
names(gendat.mouse.rs) <- c("mouse","gt1","gt2","gt3")

# Reshape tokmus for clustering/MDS
gendat.tokmouse$ID <- as.factor(paste(gendat.tokmouse$consonant,gendat.tokmouse$speaker,gendat.tokmouse$vowel,gendat.tokmouse$token))
tokmouse_cast <- cast(gendat.tokmouse,mouse~ID,value="meancx")

# Save to csv
write.csv(tokmouse_cast[,-1],file="/Users/Jonny/Documents/tok_cx.csv")
write.csv(tokmouse_cast[,1],file="/Users/Jonny/Documents/mousenames.csv")

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

# Gen vs. Base regression
base.gen2<- lm(gt2 ~ gt1,data=gendat.mouse.rs)
base.gen3 <- lm(gt3 ~ gt1,data=gendat.mouse.rs)

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
  
  ggsave(paste(basedir,"ts_area_",as.numeric(as.POSIXct(Sys.time())),".png",sep=""),plot=p.area,device="png",width=13.33,height=7.5,units="in",dpi=700,bg="transparent")
  
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
  
  ggsave(paste(basedir,"ts_step_",as.numeric(as.POSIXct(Sys.time())),".png",sep=""),plot=stepplot,device="png",width=9,height=7,units="in",dpi=700)
  
}


# Barplot split by mouse
#plot bars split by mouse
limits <- aes(ymax=cihi,ymin=cilo)
dodge <- position_dodge(width=.9)
gen.barmouse <- ggplot(gendat.mouse,aes(as.factor(mouse),meancx,fill=as.factor(gentype))) + 
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
        #axis.text.y = element_blank(),
        axis.title.x = element_text(size=rel(1.5),color="white"),        
        axis.text.x = element_text(size=rel(1.5),color="white"),
        #legend.position = c(.4,.85),
        legend.position = "none",
        legend.text = element_text(size=rel(1.2)),
        legend.title = element_text(size=rel(1.5)))
gen.barmouse
gendat.mouse <- data.frame(1)
gendat.mouse$mouse <- 'Classifier'
gendat.mouse$gentype <- 1
gendat.mouse$meancx <- .78
gendat.mouse <- gen.barmouse[1]

gen.barmouse <- ggplot(gendat.mouse,aes(x=as.factor(mouse),y=meancx,fill=as.factor(gentype))) + 
  geom_bar(position="dodge",stat="identity") +
  geom_errorbar(limits,position="dodge",width=0.25,size=0.4,color="white") + 
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
gen.barmouse

#ggsave("~/Documents/speechPlots/25_27_28_genbarmouse.svg",plot=gen.barmouse,device="svg",width=8,height=4,units="in")
ggsave(paste(basedir,"genbar_mouse_",as.numeric(as.POSIXct(Sys.time())),".png",sep=""),plot=gen.barmouse,device="png",width=9.5,height=6.5,units="in",dpi=700,bg="transparent")
ggsave(paste(basedir,"genbar_mouse_",as.numeric(as.POSIXct(Sys.time())),".png",sep=""),plot=gen.barmouse,device="png",width=1,height=6.5,units="in",dpi=700,bg="transparent")


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
ggsave(paste(basedir,"genbar_all_",as.numeric(as.POSIXct(Sys.time())),".png",sep=""),plot=gen.bar,device="png",width=3,height=6.5,units="in",dpi=700,bg="transparent")

scat_cols <- c("GT2_LINE"="#A54601","GT3_LINE"="#196B52")
# Generalization Type Scatter/Regression plot
g.genscat_scat <- ggplot(gendat.mouse.rs,aes(x=gt1)) + 
  geom_point(aes(y=gt2),col="#FF6B02",alpha=0.8,size=2) +
  geom_smooth(aes(y=gt2,colour="GT2_LINE"),method=lm,se=TRUE,size=0.75) +
  geom_point(aes(y=gt3),col="#1b9e77",alpha=0.8,size=2) +
  geom_smooth(aes(y=gt3,colour="GT3_LINE"),method=lm,se=TRUE,size=0.75) +
  scale_x_continuous(expand = c(0, 0),
                     breaks=c(0.65,0.7,0.75,0.8,0.85),labels=c("65","70","75","80","85")) + 
  scale_y_continuous(expand = c(0, 0),
                     breaks=c(0.5,0.55,0.6,0.65,0.7,0.75),labels=c("50","55","60","65","70","75")) + 
  expand_limits(x=c(0.62,0.86),y=c(0.5,0.75))+
  ylab("Generalization Accuracy (%)") +
  xlab("Learned Token Accuracy (%)") +
  scale_colour_manual(values=scat_cols,labels=c(expression(paste("Type 2 -  ",beta,": 0.52,  ", R^{2},":0.70")),expression(paste("Type 3 -  ",beta,": 0.65,  ", R^{2},":0.58"))))+
  guides(colour=guide_legend(override.aes=list(size=5)))+
  theme(panel.background = element_rect(fill="#F5F5F5"), 
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        plot.background = element_rect(fill="transparent",colour=NA),
        axis.title.y = element_text(size=rel(1),margin=margin(2,7,2,10,unit="pt")),
        axis.text.y = element_text(size=rel(1),margin=margin(1,3,1,1,unit="pt")),
        axis.text.x = element_text(size=rel(1),margin=margin(7,1,1,1,unit="pt")),
        axis.title.x = element_text(size=rel(1),margin=margin(7,2,10,2,unit="pt")),
        #axis.ticks.x = element_blank(),
        #legend.position = c(.5,.85),
        legend.position = c(.7,.2),
        legend.text = element_text(size=rel(.8)),
        legend.title = element_blank(),
        legend.background = element_blank(),
        legend.key = element_blank(),
        plot.margin = unit(c(0,0,0,0),"lines"))

g.genscat_boxhorz <- ggplot(gendat.mouse.rs,aes(x=factor(1),y=gt1))+
  geom_boxplot()+
  geom_point(size=1) + 
  scale_y_continuous(expand = c(0, 0)) + 
  expand_limits(y=c(0.62,0.86))+
  coord_flip()+
  theme(panel.background = element_blank(), 
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        plot.background = element_rect(fill="transparent",colour=NA),
        axis.title.y = element_blank(),
        axis.text.y = element_blank(),
        axis.title.x = element_blank(),
        axis.text.x = element_blank(),
        axis.ticks.x = element_blank(),
        axis.ticks.y = element_blank(),
        #legend.position = c(.5,.85),
        legend.position = "none",
        plot.margin = unit(c(1,0,0,0),"lines"))

g.genscat_boxvert <- ggplot(gendat.mouse.rs,aes(x=factor(1))) +
  geom_boxplot(aes(y=gt2),col="#A54601") +
  geom_point(aes(y=gt2),col="#FF6B02",size=1) +
  geom_boxplot(aes(x=factor(2),y=gt3),col="#196B52") +
  geom_point(aes(x=factor(2),y=gt3),col="#1b9e77",size=1) +
  scale_y_continuous(expand = c(0, 0)) + 
  expand_limits(y=c(0.5,0.75))+
  theme(panel.background = element_blank(), 
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        plot.background = element_rect(fill="transparent",colour=NA),
        axis.title.y = element_blank(),
        axis.text.y = element_blank(),
        axis.title.x = element_blank(),
        axis.text.x = element_blank(),
        axis.ticks.x = element_blank(),
        axis.ticks.y = element_blank(),
        #legend.position = c(.5,.85),
        legend.position = "none",
        plot.margin = unit(c(0,0,0,0),"lines"))

# Combine plots, see https://www.r-bloggers.com/scatterplot-with-marginal-boxplots/
g.genscat_gtabscat <- ggplotGrob(g.genscat_scat)
g.genscat_gtabhorz <- ggplotGrob(g.genscat_boxhorz)
g.genscat_gtabvert <- ggplotGrob(g.genscat_boxvert)
#g.genscat_gtabhorz <- ggplot_gtable(ggplot_build(g.genscat_boxhorz))
#g.genscat_gtabvert <- ggplot_gtable(ggplot_build(g.genscat_boxvert))

maxW <- unit.pmax(g.genscat_gtabscat$widths[2:3],g.genscat_gtabhorz$widths[2:3])
maxH <- unit.pmax(g.genscat_gtabscat$heights[4:5],g.genscat_gtabvert$heights[4:5])
g.genscat_gtabscat$widths[2:3] <- as.list(maxW)
g.genscat_gtabhorz$widths[2:3] <- as.list(maxW)
g.genscat_gtabscat$heights[4:5] <- as.list(maxH)
g.genscat_gtabvert$heights[4:5] <- as.list(maxH)

g.genscat_gtab <- gtable(widths=unit(c(7,1),"null"),height=unit(c(1,7),"null"))
g.genscat_gtab <- gtable_add_grob(g.genscat_gtab,g.genscat_gtabscat,2,1)
g.genscat_gtab <- gtable_add_grob(g.genscat_gtab,g.genscat_gtabhorz,1,1)
g.genscat_gtab <- gtable_add_grob(g.genscat_gtab,g.genscat_gtabvert,2,2)

grid.newpage()
grid.draw(g.genscat_gtab)

ggsave(paste(basedir,"genscat_",as.numeric(as.POSIXct(Sys.time())),".png",sep=""),plot=g.genscat_gtab,device="png",width=5,height=4,units="in",dpi=700,bg="transparent")

# Generalization type scatter/line combined plot
g.gensl <- ggplot(gendat.mouse,aes(x=as.factor(gentype),y=meancx,group=as.factor(mouse)))+
  geom_point(size=1)+
  geom_line(size=0.3)+
  xlab("Generalization Type")+
  ylab("Mean Accuracy (%)")+
  scale_x_discrete(expand=c(0.1,0.1))+
  scale_y_continuous(expand=c(0,0),
                     limits=c(0.5,0.86),
                     breaks=c(0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85),
                     labels=c("50","55","60","65","70","75","80","85"))+
  theme(panel.background = element_blank(), 
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        axis.text.x = element_text(size=rel(1),margin=margin(7,1,1,1,unit="pt")),
        axis.title.x = element_text(size=rel(1),margin=margin(7,2,10,2,unit="pt")),
        plot.background = element_rect(fill="transparent",colour=NA),
        #legend.position = c(.5,.85),
        legend.position = "none",
        plot.margin = unit(c(1,0,0,0),"lines"))
g.gensl
ggsave(paste(basedir,"gensl_",as.numeric(as.POSIXct(Sys.time())),".png",sep=""),plot=g.gensl,device="png",width=2,height=4,units="in",dpi=700,bg="transparent")

#Combination plot of above two
g.gensl.gt <- ggplotGrob(g.gensl)

g.combo <- grid.arrange(g.gensl.gt,g.genscat_gtab,widths=c(2,5))

ggsave(paste(basedir,"genscat_combo_",as.numeric(as.POSIXct(Sys.time())),".png",sep=""),plot=g.combo,device="png",width=8,height=5.28,units="in",dpi=700,bg="transparent")


g.genscat_gtab <- gtable_add_grob(g.genscat_gtab,g.genscat_gtabvert,2,2)


#Simple Timeseries line plot
g.tsl <- ggplot(ts.dec,aes(x=trialnum,y=rm,colour=mouse))+
  geom_line()
g.tsl

# Heatmap
g.heat <- ggplot(gendat.speaker,aes(x=as.numeric(vowel),y=as.numeric(speaker),fill=as.numeric(meancx))) + 
  geom_tile() +
  #scale_fill_gradient(low="#FFFFFF",high="#000000")
  scale_fill_distiller(palette="Greys")
g.heat


# PC on token
norm01 <- function(x){(x-min(x))/(max(x)-min(x))}

pmc <- prcomp(gendat.tokmouse_cast[,-1], center=TRUE,scale=TRUE)
pmc.x <- as.data.frame(pmc$x)
pmc.x$mouse <- tokmouse_cast$mouse
pmc.x$meancx <- (norm01(gendat.bias$meancx)+0.1)*10
pmc.x$bias <- norm01(gendat.bias$absgenbias)
g.pc <- ggplot(data=pmc.x,aes(x=PC1,y=PC2,fill=bias))+
  geom_point(size=pmc.x$meancx,stroke=1,shape=21)+
  scale_fill_gradient(low="#000000",high="#91CDFF")+
  theme(
    legend.position=c(0.75,0.5)
  )
g.pc
ggsave(paste(basedir,"pc_",as.numeric(as.POSIXct(Sys.time())),".png",sep=""),plot=g.pc,device="png",width=5,height=5,units="in",dpi=700,bg="transparent")


pmc.melt <- melt(pmc$rotation[,1:2])
pmc.melt$consonant <- as.factor(rep(gendat.token$consonant,2))
pmc.melt$speaker <- as.factor(rep(gendat.token$speaker,2))
pmc.melt$vowel <- as.factor(rep(gendat.token$vowel,2))
#pmc.melt[,1:5] <- pmc.melt[,c(1,2,4,5,3)]
#names(pmc.melt) <- c("num","PC","speaker","vowel","val")
pmc.melt <- ddply(pmc.melt,.(X2,consonant,speaker,vowel),summarise,mean.val = mean(value))
#pmc.melt <- rev(pmc.melt)
pmc.melt$ID <- factor(paste(pmc.melt$consonant,pmc.melt$speaker,pmc.melt$vowel),ordered=TRUE)

pmc.melt_g <- pmc.melt[pmc.melt$consonant==1,]
pmc.melt_g$ID <- factor(paste(pmc.melt_g$consonant,pmc.melt_g$speaker,pmc.melt_g$vowel),ordered=TRUE)
pmc.melt_b <- pmc.melt[pmc.melt$consonant==2,]
pmc.melt_b$ID <- factor(paste(pmc.melt_b$consonant,pmc.melt_b$speaker,pmc.melt_b$vowel),ordered=TRUE)


g.pcw <- ggplot(data=pmc.melt,aes(x=ID,y=mean.val,fill=as.factor(speaker)))+
  geom_bar(stat='identity')+
  coord_flip()+
  scale_x_discrete(limits=rev(levels(pmc.melt$ID)))+
  facet_wrap(~X2)+
  theme(
    #axis.text=element_blank(),
    legend.position="none",
    axis.title = element_blank()
    #axis.line = element_blank(),
    #axis.ticks = element_blank()
  )
g.pcw

g.pcw_g <- ggplot(data=pmc.melt_g,aes(x=ID,y=mean.val,fill=as.factor(speaker)))+
  geom_bar(stat='identity')+
  coord_flip()+
  scale_x_discrete(limits=rev(levels(pmc.melt_g$ID)))+
  scale_y_continuous(limits=c(-0.1263236,0.1263236))+
  facet_wrap(~X2)+
  theme(
    axis.text=element_blank(),
    legend.position="none",
    axis.title = element_blank(),
    axis.line = element_blank(),
    axis.ticks = element_blank(),
    strip.text = element_blank(),
    strip.background = element_blank(),
    plot.margin = margin(0,0,0,0,unit="npc"),
    panel.spacing.y = unit(0,"npc")
  )
g.pcw_g

g.pcw_b <- ggplot(data=pmc.melt_b,aes(x=ID,y=mean.val,fill=as.factor(speaker)))+
  geom_bar(stat='identity')+
  coord_flip()+
  scale_x_discrete(limits=rev(levels(pmc.melt_b$ID)))+
  scale_y_continuous(limits=c(-0.1263236,0.1263236))+
  facet_wrap(~X2)+
  theme(
    axis.text=element_blank(),
    legend.position="none",
    axis.title = element_blank(),
    axis.line = element_blank(),
    axis.ticks = element_blank(),
    strip.text = element_blank(),
    strip.background = element_blank(),
    plot.margin = margin(0,0,0,0,unit="npc"),
    panel.spacing.y = unit(0,"npc")
  )
g.pcw_b

# Heatmap of each mouse's performance on each token
gendat.tokmus$mouse <- as.factor(gendat.tokmus$mouse)
gendat.tokmus_g <- gendat.tokmus[gendat.tokmus$consonant==1,]
gendat.tokmus_b <- gendat.tokmus[gendat.tokmus$consonant==2,]

#Reorder factor so it plots right...
gendat.tokmus_g$vowel <- factor(gendat.tokmus_g$vowel,ordered=TRUE)
gendat.tokmus_g$vowel <- factor(gendat.tokmus_g$vowel,levels(gendat.tokmus_g$vowel)[c(6,5,4,3,2,1)])
gendat.tokmus_b$vowel <- factor(gendat.tokmus_b$vowel,ordered=TRUE)
gendat.tokmus_b$vowel <- factor(gendat.tokmus_b$vowel,levels(gendat.tokmus_b$vowel)[c(6,5,4,3,2,1)])

g.tok_g <- ggplot(gendat.tokmus_g,aes(x=mouse,y=vowel,fill=meancx)) + 
  geom_tile() + 
  ylab("Vowel") + 
  scale_y_discrete(labels=c("/u/","/e/","/ae/","/a/","o","/I/")) + # reverse order to make the plot work
  scale_fill_distiller(type="div",palette="RdGy",limits=c(0,1),
                       breaks=c(0,0.5,1),labels=c("0%","50%","100%"),
                       name="Accuracy")+
  facet_grid(speaker ~ .,scales="free_y",space="free_y") +
  theme(panel.spacing=unit(-0.008,"npc"),
        axis.line=element_blank(),
        axis.text.x = element_blank(),
        axis.ticks.x = element_blank(),
        axis.title.x = element_blank(),
        axis.title.y = element_blank(),
        axis.text.y=element_text(size=rel(0.5)),
        legend.position="top",
        legend.margin=margin(c(0,0.12,-0.017,0.4),unit = "npc"),
        legend.text = element_text(size=rel(0.5)),
        legend.title=element_blank(),
        legend.key.height = unit(0.01,"npc"),
        strip.text = element_text(size=rel(0.6)),
        plot.margin = margin(0.04,0,0,0,unit="npc"))
g.tok_gtab <- ggplotGrob(g.tok_g) # Add consonant panel
g.tok_gtab <- gtable_add_cols(g.tok_gtab,unit(g.tok_gtab$widths[[5]],'cm'),7)
g.tok_gtab <- gtable_add_grob(g.tok_gtab,
                              rectGrob(gp=gpar(col=NA,fill=gray(0.5))),
                              6,8,16,name="glab")
g.tok_gtab <- gtable_add_grob(g.tok_gtab,
                              textGrob("/g/",rot=-90,gp=gpar(col=gray(1))),
                              6,8,16,name="gtext")

g.tok_b <- ggplot(gendat.tokmus_b,aes(x=mouse,y=vowel,fill=meancx)) + 
  geom_tile() + 
  ylab("Vowel") + 
  scale_y_discrete(labels=c("/u/","/e/","/ae/","/a/","o","/I/")) +
  scale_fill_distiller(type="div",palette="RdGy",limits=c(0,1))+
  xlab("Mouse") +
  facet_grid(speaker ~ .,scales="free_y",space="free_y") +
  theme(panel.spacing=unit(-0.008,"npc"),
        axis.line=element_blank(),
        axis.ticks.x=element_blank(),
        axis.text.x=element_blank(),
        axis.title.y=element_blank(),
        axis.text.y=element_text(size=rel(0.5)),
        axis.title.x=element_text(size=rel(0.7),margin=margin(0,0,0,0,unit="pt")),
        legend.position="none",
        strip.text=element_text(size=rel(0.6)),
        plot.margin=margin(0,0,0.04,0,unit="npc"))
g.tok_btab <- ggplotGrob(g.tok_b) # Add consonant panel
g.tok_btab <- gtable_add_cols(g.tok_btab,unit(g.tok_btab$widths[[5]],'cm'),7)
g.tok_btab <- gtable_add_grob(g.tok_btab,
                              rectGrob(gp=gpar(col=NA,fill=gray(0.5))),
                              4,8,14,name="glab")
g.tok_btab <- gtable_add_grob(g.tok_btab,
                              textGrob("/b/",rot=-90,gp=gpar(col=gray(1))),
                              4,8,14,name="gtext")

grid.newpage()
g.tokenplot <- grid.arrange(g.tok_gtab,g.tok_btab,ncol=1,heights=c(1,1))

ggsave(paste(basedir,"tok_heatmap_",as.numeric(as.POSIXct(Sys.time())),".png",sep=""),plot=g.tokenplot,device="png",width=3,height=5,units="in",dpi=700,bg="transparent")

# Combine heatmap and PCplots
g.pcw_gtab <- ggplotGrob(g.pcw_g)
g.pcw_btab <- ggplotGrob(g.pcw_b)

g.tok_gtab2 <- g.tok_gtab
g.tok_btab2 <- g.tok_btab

g.tok_gtab2 <- gtable_add_cols(g.tok_gtab2,rep(unit(0.15,"npc"),2),0)
g.tok_btab2 <- gtable_add_cols(g.tok_btab2,rep(unit(0.15,"npc"),2),0)

g.tok_gtab2 <- gtable_add_grob(g.tok_gtab2,
                               g.pcw_gtab,
                               6,1,17,2,name="gtok")

g.tok_btab2 <- gtable_add_grob(g.tok_btab2,
                               g.pcw_btab,
                               4,1,15,2,name="gtok")

g.tok_gtab2 <- gtable_add_grob(g.tok_gtab2,
                               rectGrob(gp=gpar(col=NA,fill=gray(0.5))),
                               5,1,5,name="pc1box")
g.tok_gtab2 <- gtable_add_grob(g.tok_gtab2,
                               rectGrob(gp=gpar(col=NA,fill=gray(0.5))),
                               5,2,5,name="pc2box")
g.tok_gtab2 <- gtable_add_grob(g.tok_gtab2,
                               textGrob("PC1",gp=gpar(col=gray(1))),
                               5,1,5,name="pc1")
g.tok_gtab2 <- gtable_add_grob(g.tok_gtab2,
                               textGrob("PC2",gp=gpar(col=gray(1))),
                               5,2,5,name="pc2")

grid.newpage()
g.tokpc_plot <- grid.arrange(g.tok_gtab2,g.tok_btab2,ncol=1)

ggsave(paste(basedir,"tok_heatmap_PC_",as.numeric(as.POSIXct(Sys.time())),".png",sep=""),plot=g.tokpc_plot,device="png",width=4,height=5,units="in",dpi=700,bg="transparent")


# Bias scatterplot/regression plot
g.bias <- ggplot(gendat.bias,aes(x=bias,y=conscx)) + 
  geom_point() +
  geom_smooth(method=lm,se=FALSE) +
  geom_point(aes(x=absbias,y=meancx))+
  geom_smooth(aes(x=absbias,y=meancx),method=lm,se=FALSE)+
  geom_point(aes(x=abs(genbias),y=gencx),col="red")+
  geom_smooth(aes(x=abs(genbias),y=gencx),se=FALSE,method=lm,col="red")+
  #geom_point(aes(x=bias,y=genconscx),col="green") +
  #geom_smooth(aes(x=bias,y=genconscx),method=lm,se=FALSE,col="green")+
  geom_point(aes(x=abs(genbias),y=cxdiff),col="green")+
  geom_smooth(aes(x=abs(genbias),y=cxdiff),method=lm,se=FALSE,col="green")+
  #geom_point(aes(x=bias,y=genbias),col="green")+
  #geom_smooth(aes(x=bias,y=genbias),method=lm,col="green")+
  #geom_point(aes(x=absbias,y=gencx),col="red")+
  #geom_smooth(aes(x=absbias,y=gencx),method=lm,se=TRUE,col="red")+
  #geom_point(aes(x=absbias,y=cxdiff),col="green") +
  #geom_smooth(aes(x=absbias,y=cxdiff),method=lm,se=TRUE,col="green")+
  ylab("Bias (%responses - %target)") +
  xlab("% Correct") +
  theme(panel.background = element_blank(), 
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        plot.background = element_rect(fill="transparent",colour=NA)
        #axis.title.y = element_text(size=rel(1),margin=margin(2,7,2,10,unit="pt")),
        #axis.text.y = element_text(size=rel(1),margin=margin(1,3,1,1,unit="pt")),
        #axis.text.x = element_text(size=rel(1),margin=margin(7,1,1,1,unit="pt")),
        #axis.title.x = element_text(size=rel(1),margin=margin(7,2,10,2,unit="pt")),
        #axis.ticks.x = element_blank(),
        #legend.position = c(.5,.85),
        #legend.position = c(.7,.2),
        #legend.text = element_text(size=rel(.8)),
        #legend.title = element_blank(),
        #legend.background = element_blank(),
        #legend.key = element_blank(),
        #plot.margin = unit(c(0,0,0,0),"lines"))
  )
g.bias

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
gendat.token$meancx <- gendat.token$meancx-0.5
gendat.speaker$meancx <- gendat.speaker$meancx-0.5
classis <- ggplot(gendat.speaker, aes(x=as.factor(ID),y=meancx,fill=factor(speaker))) +
  geom_bar(stat='identity',position="dodge") +
  scale_y_continuous(limits = c(-.25,.5))+
  #scale_fill_brewer(palette="Set1") +
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
ggsave(paste(basedir,"tokclass",as.numeric(as.POSIXct(Sys.time())),".png",sep=""),plot=classis,device="png",width=7,height=6.5,units="in",dpi=700,bg="transparent")


