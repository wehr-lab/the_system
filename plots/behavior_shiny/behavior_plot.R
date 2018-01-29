library(shiny)
library(tidyverse)
library(ggplot2)
library(zoo)

# Load behavior data
active_mice <- c('7926','7940','7941','7942','7943','7944',
                 '7961','8046','8075','8076','8077','8078',
                 '8104','8105','8106')

data_dir <- "~/Documents/speechData/"

remove(all_tab)
for(m in active_mice){
  filename <- paste(data_dir, m, '.csv', sep="")
  tab <- read.csv(filename)
  tab$mouse <- m
  if(exists("all_tab")){
    all_tab <- rbind(all_tab, tab)
  } else {
    all_tab <- tab
  }
}
all_tab$mouse <- as.factor(all_tab$mouse)
all_tab <- all_tab[!is.na(all_tab$correct),]

# shiny ui
# define defaults
n_days <- 5000
window <- 100


ui <- fluidPage(
  titlePanel("Mouse Behavior Test"),
  fluidRow(
    column(6, numericInput('n_days', "Number of Days", n_days)),
    column(6, numericInput('window', "Window Length", window))
    ),
  plotOutput('plot1', height="100%")
)


server <- shinyServer(function(input, output){
  # subset
  all_tab_sub <- reactive({
    all_tab %>%
      group_by(mouse) %>%
      arrange(trialNumber) %>%
      filter(trialNumber >= max(trialNumber)-input$n_days-input$window) %>%
      mutate(rmeancx = rollmean(correct, input$window, fill=0, align="right"))
  })
  
    
  
  output$plot1 <- renderPlot({
    p <- ggplot(data=all_tab_sub(), aes(x=trialNumber))+
    #geom_line(aes(y=rollmean(correct, input$window, na.pad=TRUE, align="right")))+
    facet_wrap(~mouse, ncol=1, scales="free_x")+
    geom_line(aes(y=rmeancx))
    return(p)
    }, 
    height=function(){
      ats = all_tab_sub()
      nplots <- length(levels(ats$mouse))
      return(nplots*150)})
})


shinyApp(ui=ui, server=server)
