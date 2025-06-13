ui<-page_sidebar(

  theme=bs_theme(bootswatch="flatly"),
  title=div(
    style="margin-left: 30px;display: flex;flex-direction: column;",
    div("Areas Of Concern",style="font-size: 24px; font-weight: bold;color: #2c3e50;"),
    tags$a("Expert-driven explainable artificial intelligence models can detect multiple climate hazards relevant for agriculture",href="https://www.nature.com/articles/s43247-024-01987-3.pdf",target="_blank",
           style = "font-size: 14px; text-underline-offset: 6px; color: #2c3e50; text-decoration: underline; margin-top: 8px;max-width: 900px;cursor: pointer;")
  ),  
  useShinyFeedback(),
  use_waiter(),


    sidebar=sidebar(
      h4("Controls"),
      sliderInput("atts","Number of training attempts",value=3,min=1,max=10),
      sliderInput("depth","Decision tree max depth",value=5,min=1,max=10),      
      selectInput("sTgt","Select the Area Of Concern:",sTgts,multiple=FALSE,selected=c("Drought")),
      pickerInput("sFeats","Select your features here:",all_feature_names,multiple = TRUE,options=list(
        `actions-box` = TRUE,
        `deselect-all-text` = "Clear All",
        `select-all-text` = "Select All",
        `none-selected-text` = "No feature selected"        
      )),
      radioButtons(inputId="metrics",label="Explainability:",choices=c("F1Score","variableImportance","SHAP"),selected="F1Score"),
      actionButton("start","Start the Analysis")      
    ),
      navset_card_tab(
        
        tabPanel("Explainability",plotOutput("pVarImp",height="600px")),
        tabPanel("Area Of Concern (probability)",plotOutput("map")),
        tabPanel("Feature Maps",sidebarLayout(
                   sidebarPanel(
                     uiOutput("feature_selector")
                   ),mainPanel(
                     plotOutput("feature",height="600px"))                             
                   ))),type=4
)
