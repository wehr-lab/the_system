# Make gendat using load_generalization function in speech_behavior_analysis

require(zoo)
require(plyr)
require(reshape)
require(binom)
require(ggplot2)
require(signal)

basedir <- "/Users/Jonny/Dropbox/Lab Self/inFormants/Analysis/Plots/"


# Make unique ID for each token
gd.ts <- gendat
gd.ts$ID <- as.factor(paste(gd.ts$consonant,gd.ts$vowel,gd.ts$speaker,gd.ts$token))
gd.ts <- gd.ts[,c("mouse","ID","gentype","correct")]

# Number the appearances of each token
gd.ts <- ddply(gd.ts,.(mouse,ID),mutate,nappear = seq(1,length(correct)))

# Reshape dataframe to get appearances by mouse/token
gd.ts.rs <- melt(gd.ts,id=c("mouse","gentype","ID","nappear"),measure.vars="correct")
gd.ts.rs2 <- cast(gd.ts.rs, mouse+gentype+ID ~ numappear)

# Get mean correct by # of appearance
gd.ts.byappear <- gd.ts
gd.ts.byappear[gd.ts.byappear$gentype==3,]$gentype <- 2
gd.ts.byappear <- ddply(gd.ts.byappear,.(gentype,nappear),summarize,meancx=mean(correct),nmice = length(unique(mouse)),cilo = binom.confint(sum(correct),length(correct),conf.level=0.95,method="exact")[[5]],cihi = binom.confint(sum(correct),length(correct),conf.level=0.95,method="exact")[[6]])
gd.ts.byappear <- gd.ts.byappear[gd.ts.byappear$nmice == 13,]
gd.ts.byappear <- gd.ts.byappear[gd.ts.byappear$nappear <= 50,]
#gd.ts.byappear <- gd.ts.byappear[(gd.ts.byappear$gentype == 2) | (gd.ts.byappear$gentype == 3),]

# Plot above
g.appear <- ggplot(gd.ts.byappear,aes(x=nappear,y=meancx))+
  #xlim(c(0,25))+
  #geom_crossbar(aes(ymin=cilo,ymax=cihi,fill=as.factor(gentype),colour=as.factor(gentype)),alpha=0.1,size=0.4,width=0.3)+
  geom_linerange(aes(ymin=cilo,ymax=cihi,colour=as.factor(gentype)),size=0.7,alpha=0.3)+
  geom_point(aes(colour=as.factor(gentype)),alpha=0.8)+
  geom_smooth(method="glm",aes(colour=as.factor(gentype)),formula=(y~log(x)),se=F)+
  labs(x = "# of Presentations",y="Mean % Correct")+
  scale_fill_brewer(palette = "Set1")+
  scale_colour_brewer(palette = "Set1",labels=c("Learned Tokens","Novel Tokens")) +
  scale_y_continuous(limits=c(0.5,0.8),breaks=c(0.5,0.6,0.7,0.8),labels = c("50","60","70","80"))+
  theme(
    panel.background = element_blank(), 
    panel.grid.minor = element_blank(),
    panel.grid.major = element_blank(),
    axis.text.x = element_text(size=rel(1.25)),
    axis.text.y = element_text(size=rel(1.25),margin=margin(t=0,r=0,b=0,l=0,unit="pt")),
    axis.title.y = element_text(size=rel(1.25)),
    axis.title.x = element_text(size=rel(1.25)),
    #axis.ticks.y = element_blank(),
    legend.key = element_blank(),
    legend.title = element_blank(),
    legend.text = element_text(size=rel(1.1)),
    legend.background = element_blank(),
    legend.position = c(0.7,0.15)
  )
g.appear
ggsave(paste(basedir,"ts_appear",as.numeric(as.POSIXct(Sys.time())),".png",sep=""),plot=g.appear,device="png",width=5,height=2.5,units="in",dpi=700,bg="transparent")


gd.ts.byappear.lm1 <- lm(meancx~log(nappear)*as.factor(gentype),data=gd.ts.byappear)
summary(gd.ts.byappear.lm1)

# By mouse
gd.ts.bymouse <- gd.ts[(gd.ts$nappear<=50) & (gd.ts$gentype == 2 | gd.ts$gentype == 3),]
gd.ts.bymouse[gd.ts.bymouse$gentype==3,]$gentype <- 2 
gd.ts.bymouse <- ddply(gd.ts.bymouse,.(mouse,nappear),summarize,meancx=mean(correct),cilo = binom.confint(sum(correct),length(correct),conf.level=0.95,method="exact")[[5]],cihi = binom.confint(sum(correct),length(correct),conf.level=0.95,method="exact")[[6]])
gd.ts.bymouse$mouse <- as.factor(gd.ts.bymouse$mouse)
g.mouse <- ggplot(gd.ts.bymouse,aes(x=nappear,y=meancx))+
  geom_line()+
  facet_grid(mouse~.)+
  #geom_line(aes(colour=as.factor(mouse)))
  geom_ribbon(aes(ymin=cilo,ymax=cihi),alpha=0.3,linetype=2,size=0.5)

g.mouse

gd.ts.just28 <- gd.ts.bymouse[gd.ts.bymouse$mouse=="6928",]

gd.ts.lm1 <- lm(meancx~nappear,data=gd.ts.bymouse)
gd.ts.lm2 <- lm(meancx~log(nappear)+mouse,data=gd.ts.bymouse)
gd.ts.lm3 <- lm(meancx~log(nappear)*mouse,data=gd.ts.bymouse)
anova(gd.ts.lm1,gd.ts.lm2,gd.ts.lm3)
summary(gd.ts.lm3)
summary(gd.ts.lm2)

gd.ts.lms <- ddply(gd.ts.bymouse,.(mouse,gentype), function(gd.ts.bymouse){
  model<-lm(meancx~log(nappear),data=gd.ts.bymouse)
  coef(model)
})
colnames(gd.ts.lms) <- c("mouse","intercept","nappear")
g.mouse.lm <- ggplot(gd.ts.bymouse,aes(x=nappear,y=meancx,colour=as.factor(gentype)))+
  geom_point()+
  geom_smooth(method="glm",aes(colour=as.factor(gentype),fill=as.factor(mouse)),formula=(y~log(x)),se=F)
  #geom_abline(data=gd.ts.lms,aes(intercept=intercept,slope=nappear,colour=mouse))
g.mouse.lm

# Plot all mice's novel learning curve

g.bymouse <- ggplot(gd.ts.bymouse,aes(x=nappear,y=meancx))+
  #xlim(c(0,25))+
  #geom_crossbar(aes(ymin=cilo,ymax=cihi,fill=as.factor(gentype),colour=as.factor(gentype)),alpha=0.1,size=0.4,width=0.3)+
  geom_linerange(aes(ymin=cilo,ymax=cihi,colour=mouse),size=0.7,alpha=0.3)+
  geom_point(aes(colour=mouse),alpha=0.8)+
  geom_smooth(method="glm",aes(colour=mouse),formula=(y~log(x)),se=F)+
  labs(x = "# of Presentations",y="Mean % Correct")+
  #scale_fill_brewer(palette = "Set1")+
  #scale_colour_brewer(palette = "Set1") +
  #scale_y_continuous(limits=c(0.5,0.8),breaks=c(0.5,0.6,0.7,0.8),labels = c("50","60","70","80"))+
  theme(
    panel.background = element_blank(), 
    panel.grid.minor = element_blank(),
    panel.grid.major = element_blank(),
    axis.text.x = element_text(size=rel(1.25)),
    axis.text.y = element_text(size=rel(1.25),margin=margin(t=0,r=0,b=0,l=0,unit="pt")),
    axis.title.y = element_text(size=rel(1.25)),
    axis.title.x = element_text(size=rel(1.25)),
    #axis.ticks.y = element_blank(),
    legend.key = element_blank(),
    legend.title = element_blank(),
    legend.text = element_text(size=rel(1.1)),
    legend.background = element_blank()
    #legend.position = c(0.7,0.15)
  )
g.bymouse


# Bytoken
gd.ts.bytok <- gendat[gendat$mouse=="7012",]
gd.ts.bytok$ID <- as.factor(paste(gd.ts.bytok$consonant,gd.ts.bytok$vowel,gd.ts.bytok$speaker,gd.ts.bytok$token))
gd.ts.bytok <- ddply(gd.ts.bytok,.(mouse,ID),mutate,nappear = seq(1,length(correct)))
gd.ts.bytok <- gd.ts.bytok[gd.ts.bytok$nappear<=50,]
gd.ts.bytok <- ddply(gd.ts.bytok,.(ID,nappear),summarize,meancx=mean(correct),cilo = binom.confint(sum(correct),length(correct),conf.level=0.95,method="exact")[[5]],cihi = binom.confint(sum(correct),length(correct),conf.level=0.95,method="exact")[[6]])
g.bytok <- ggplot(gd.ts.bytok,aes(x=nappear,y=ID))+
  geom_tile(aes(fill=meancx))+
  theme(
    legend.position="none"
  )
g.bytok
gd.ts.tok.lm1 <- glm(meancx~log(nappear),data=gd.ts.bytok,family="binomial")
gd.ts.tok.lm2 <- glm(meancx~log(nappear)+log(nappear):ID,data=gd.ts.bytok,family="binomial")
gd.ts.tok.lm3 <- glm(meancx~log(nappear)*ID,data=gd.ts.bytok,family="binomial")
anova(gd.ts.tok.lm1,gd.ts.tok.lm2,gd.ts.tok.lm3)
summary(gd.ts.tok.lm1)
summary(gd.ts.tok.lm2)


#######
ts.dec <- ts.dec[ts.dec$step==15,]
ts.dec <- ts.dec[!(ts.dec$mouse %in% c("6924","6925","6926","6964","6927","6928","6960","6965","6966","6967","7007","7012")),]
ts.dec.1m <- ts.dec[(ts.dec$mouse=="6927"),]
ts.dec.1m <- ts.dec.1m[(ts.dec.1m$session >=  96 & ts.dec.1m$session<=98),]

g.ts <- ggplot(ts.dec.1m,aes(x=trialnum,y=rm))+
  geom_point(aes(colour=as.factor(session)))+
  facet_grid(mouse~.,scales="free")
g.ts

