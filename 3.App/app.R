rm(list=objects())
source("global.R")
source("ui.R")
source("server.R")

# Run the app
shinyApp(ui = ui, server = server)