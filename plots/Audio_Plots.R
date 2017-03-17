library(audio)
library(ggplot2)
a <- load.wave("/Users/Jonny/Dropbox/Lab Self/inFormants/Presentations/FYP/herzog_penguins.wav")
a <- data.frame(unclass(a))
names(a) <- "samples"

g <- ggplot(a,aes(x=1:nrow(a),y=samples)) +
  geom_line(size=.05) + 
  theme(panel.background = element_rect(fill="transparent",colour=NA), 
        plot.background = element_rect(fill="transparent",colour=NA), 
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        axis.title.y = element_blank(),
        axis.text.y = element_blank(),
        axis.title.x = element_blank(),
        axis.text.x = element_blank(),
        axis.ticks = element_blank())
ggsave(filename="herzog_penguins.png",plot=g,path="/Users/Jonny/Dropbox/Lab Self/inFormants/Presentations/FYP",width=8,height=1,dpi=1000,bg="transparent")

