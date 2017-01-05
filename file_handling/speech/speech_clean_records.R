spfiles <- c("/Users/Jonny/Documents/speechData/6924.csv",
             "/Users/Jonny/Documents/speechData/6925.csv",
             "/Users/Jonny/Documents/speechData/6926.csv",
             "/Users/Jonny/Documents/speechData/6927.csv",
             "/Users/Jonny/Documents/speechData/6928.csv",
             "/Users/Jonny/Documents/speechData/6960.csv",
             "/Users/Jonny/Documents/speechData/6964.csv",
             "/Users/Jonny/Documents/speechData/6965.csv",
             "/Users/Jonny/Documents/speechData/6966.csv",
             "/Users/Jonny/Documents/speechData/6967.csv",
             "/Users/Jonny/Documents/speechData/7007.csv",
             "/Users/Jonny/Documents/speechData/7012.csv",
             "/Users/Jonny/Documents/speechData/7058.csv")


fix_dir = "/Users/Jonny/Documents/fixSpeechData/"

keep_cols <- c("consonant","speaker","vowel","token","correct","gentype","step","session","date","response","target")



fix_generalization <- function(spfiles=spfiles,keep_cols=keep_cols){
  gendat <- data.frame(setNames(replicate(length(keep_cols)+1,numeric(0),simplify = F),keep_cols))
  prog_bar <- txtProgressBar(min=0,max=length(spfiles),width=20,initial=0,style=3)
  i=0
  # Loop through files
  for (f in spfiles){
    i=i+1
    setTxtProgressBar(prog_bar,i)
    
    sp <- read.csv(f)
    sp.gens <- subset(sp,((step>=5) & (!is.na(correct)) & (!is.na(response))),select=keep_cols)
    
    
    # Get mouse name & add to dataframe
    mname <- substr(f,35,38)
    sp.gens$mouse <- mname
    
    # Fix speaker # for mice that use the new stimmap
    if ((mname == "7007")|(mname == "7012")|(mname == "7058")){
      sp.gens.temp <- sp.gens #Make a copy so we don't run into recursive changes
      sp.gens.temp[(sp.gens$speaker == 1) & (sp.gens$step >= 5),]$speaker <- 5
      sp.gens.temp[(sp.gens$speaker == 2) & (sp.gens$step >= 5),]$speaker <- 4
      sp.gens.temp[(sp.gens$speaker == 3) & (sp.gens$step >= 5),]$speaker <- 1
      sp.gens.temp[(sp.gens$speaker == 4) & (sp.gens$step >= 5),]$speaker <- 2
      sp.gens.temp[(sp.gens$speaker == 5) & (sp.gens$step >= 5),]$speaker <- 3
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
    
    # Return speaker ID to original mapping
    if ((mname == "7007")|(mname == "7012")|(mname == "7058")){
      sp.gens.temp <- sp.gens
      sp.gens.temp[(sp.gens$speaker == 5) & (sp.gens$step >= 5),]$speaker <- 1
      sp.gens.temp[(sp.gens$speaker == 4) & (sp.gens$step >= 5),]$speaker <- 2
      sp.gens.temp[(sp.gens$speaker == 1) & (sp.gens$step >= 5),]$speaker <- 3
      sp.gens.temp[(sp.gens$speaker == 2) & (sp.gens$step >= 5),]$speaker <- 4
      sp.gens.temp[(sp.gens$speaker == 3) & (sp.gens$step >= 5),]$speaker <- 5
      sp.gens <- sp.gens.temp
    }
    
    #prevent overlapping gentypes from before sampling was not mutually exclusive.
    # First knock everything not a 3 to a 2
    try(sp.gens[sp.gens$date<736589 & (sp.gens$vowel==3|sp.gens$vowel==2|sp.gens$vowel==1),]$gentype <- 2,silent = T) 
    # Then specifically define all the 1's.
    try(sp.gens[sp.gens$date<736589 & (sp.gens$speaker==1|sp.gens$speaker==2) & (sp.gens$vowel==1|sp.gens$vowel==2) & (sp.gens$token==1|sp.gens$token==2),]$gentype <- 1,silent = T)
    try(sp.gens[sp.gens$date<736589 & (sp.gens$speaker==1|sp.gens$speaker==2) & sp.gens$vowel==3 & sp.gens$token==1,]$gentype <- 1,silent = T)
    
    # Renumber target & response to be 0 and 1 rather than 1 and 3
    sp.gens[sp.gens$target == 1,]$target <- 0
    sp.gens[sp.gens$target == 3,]$target <- 1
    sp.gens[sp.gens$response == 1,]$response <- 0
    sp.gens[sp.gens$response == 3,]$response <- 1
    
    # Idiosyncratic fixes for each mouse (run on wrong day, data exclusion criteria, etc.)
    if (mname == "7007"){
      sp.gens <- sp.gens[!sp.gens$session==115,] # 52 run on file
    }
    
    
    write.csv(sp.gens,file=paste(fix_dir,mname,'.csv',sep=""))
  }
}