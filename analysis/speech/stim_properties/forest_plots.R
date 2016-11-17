require(ggplot2)
require(reshape2)
require(matlab)

contrib <- read.csv('/Users/Jonny/Documents/tempnpdat/contrib_avg.csv',header=FALSE)
contrib <- as.data.frame(lapply(contrib, normalize))
contrib2 <- contrib
contrib$yax <- factor(rownames(contrib),levels=rownames(contrib)[order(rownames(contrib))])
contrib_melt <- melt(contrib,id.vars="yax",value.name="contrib",variable.name='xax')
contrib_melt$yax <- as.numeric(contrib_melt$yax)
contrib_melt$xax <- as.numeric(contrib_melt$variable)

ggplot(data=contrib_melt,aes(x=xax,y=yax))+
  geom_raster()
  scale_colour_gradient(limits=c(min(contrib2),max(contrib2)))


normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}

contrib <- read.csv('/Users/Jonny/Documents/tempnpdat/contrib_avg.csv',header=FALSE,row.names=1)
contrib <- matrix(contrib,ncol=199,nrow=13,byrow=FALSE)
imagesc(x=seq(ncol(contrib)),y=seq(nrow(contrib)),C=contrib,xlab="text",ylab="text",col=jet.color(16))
