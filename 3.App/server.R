server<-function(input,output,session){

  NROUNDS<-50
  
  # Ensemble Members
  dfRes = list()
  dfResTrn = list()
  dfResVal = list()
  dfResTst = list()
  errMetrics = list()
  dfVarImp = list()
  dfSHAP = list()
  dfSHAP[[1]] = list()
  dfSHAP[[2]] = list()
  
  
  #Names for showing the map of selected driver
  sVarImps = c("msl_mean", "sp_mean", "t2m_mean", "tmax_mean", "tmin_mean", "tp_mean", "z500_mean", 
               "msl_anom", "sp_anom", "t2m_anom", "tmax_anom", "tmin_anom", "tp_anom", "z500_anom", 
               "spi01", "spi03", "spi06", "spi09", "spi12", "smant", "fpanv", 
               "daystmax25a30C", "daystmax30C", "daystmin0C", "daystminm10C", "daystminm18C", "daystp01mm", "daystp30mm")  
    
  file_ready_signal<-reactiveVal(NULL)
  file_ready_signal_pred<-reactiveVal(NULL)
  dfRes_finale<-reactiveVal(NULL)
  dfVarImpNorm<-reactiveVal(NULL)
  dfSHAP_react<-reactiveVal(NULL)

  w <- Waiter$new(
    html = tagList(div(spin_fading_circles(),style="color: black"), h3("Working... please wait",style="color: black")),
    color = "#ffffffcc"  # light transparent overlay
  )

  eventReactive(input$start,{
    input$atts
  })->atts
  
  eventReactive(input$start,{
    input$depth
  })->MAX_DEPTH  
  
  #Data for training
  eventReactive(input$start,{
    w$show()
    read_delim("dfDatabase_sample.csv",delim=",",col_names = TRUE)
  })->dfData
  
  #Data for prediction
  eventReactive(input$start,{
    read_delim("dfDatabaseProd_sample.csv",delim=",",col_names=TRUE) 
  })->dfData_pred
  
  eventReactive(dfData(),{
    
    req(dfData())
    
    # Getting dates where AOC exist
    sort(unique(dfData()$Time[rowSums(dfData()[names(dfData()) %in% isolate(input$sTgt)], na.rm=T) > 0]))
    
  })->dfDates
  
  
  observeEvent(input$start,{
    
    # Creating output folder directory
    if (dir.exists(isolate(input$sTgt))) { unlink(isolate(input$sTgt), recursive=TRUE) }
    dir.create(isolate(input$sTgt))
    
  })
  
  eventReactive(input$start,{
    
    if(length(input$sFeats)<2){
      NULL
    }else{
      input$sFeats
    }
    
  })->sfeats
  
  observeEvent(input$start,{
    
    if(length(input$sFeats)<2){
      showFeedbackWarning(
        inputId="sFeats",
        text="Please select at least two features"
      )
    }else{
      hideFeedback("sFeats")
    }
    
  })
  
  reactive({
    
    # Creating the training database
    req(sfeats())

    na.omit(dfData()[,names(dfData()) %in% c(sIDs, isolate(sfeats()), isolate(input$sTgt)),drop=FALSE])->tmp

    # Keeping only dates where the AOC exist
    tmp[tmp$Time %in% dfDates(),]
    
  })->dfDataset 
    
  reactive({  
    
    req(dfDataset())

    idT = which(dfDataset()[[isolate(input$sTgt)]]==1) 
    idT_dates = sort(unique(dfDataset()[idT,]$Time))
    tmp = dfDataset()[dfDataset()$Time %in% idT_dates,]
    # Shuffling the data and getting the IDs
    tmp = tmp[sample(NROW(tmp)),]
    dfIDs = tmp[,names(tmp) %in% c(sIDs)] 
    
    # Preparing the dfDatasetTgt object
    dfDatasetTgt = tmp[,names(tmp) %in% c(isolate(sfeats()),isolate(input$sTgt)),drop=FALSE]
    names(dfDatasetTgt)[names(dfDatasetTgt)==isolate(input$sTgt)] = "y"
   
    dfDatasetTgt
    
  })->dfDatasetTgt
  
  
  # Preparing the data for a xgb model
  reactive({
    
    req(dfDatasetTgt())
    print(dfDatasetTgt())    
    
    xgb.DMatrix(data=as.matrix(subset(dfDatasetTgt(), select=-c(y))), label=dfDatasetTgt()$y)
  })->dfXGB
  
  reactive({

    req(dfDatasetTgt())    
    
    Matrix::Matrix(as.matrix(subset(dfDatasetTgt(), select=-c(y))), sparse=T)
  })->dfSHAPsm  
  
  
  # reactive({
  #   #req(dfDatasetTgt())
  #   req(shapExp())
  #   (sample(1:nrow(shapExp$shap_contrib),1000))
  #   #sample(1:dim(dfDataset())[1], 1000)
  # })->shap_idx
  
  
  observeEvent(dfDatasetTgt(),{

    shap_idx<-c()    
    for (att in 1:isolate(input$atts)) {
      
      # Training block of the model ==============================================
      paste0("  Processing fold ", att, " out of ", isolate(input$atts))
      # Specifying the % train-validation-test split
      trn_idx = as.vector(createDataPartition(dfDataset()[[1]][sample(NROW(dfDataset()))], p=0.50, list=FALSE))
      tst_idx = seq(1,NROW(dfDataset()),1)[-trn_idx]
      # Specifying the % train-validation-test split
      trn_idx = seq(1,NROW(dfDataset()),1)[-tst_idx]
      val_idx = createDataPartition(trn_idx, p=0.25, list=FALSE); trn_idx = trn_idx[-val_idx]
      
      # Creating the training, validation and test sets
      dfTrn = dfDatasetTgt()[trn_idx,,drop=FALSE]
      dfVal = dfDatasetTgt()[val_idx,,drop=FALSE]
      dfTst = dfDatasetTgt()[tst_idx,,drop=FALSE]
    
      # Balancing the classes 0 and 1
      idT = which(dfTrn$y==1)
      idF = which(dfTrn$y==0)
      idF = sample(idF, length(idT))
      dfTrn = dfTrn[sort(c(idF, idT)),]
      scale_pos_weight = sum(dfTrn$y==0)/sum(dfTrn$y==1)
      
      # Preparing the data for a xgb model
      dfTrnXGB = xgb.DMatrix(data=as.matrix(subset(dfTrn, select=-c(y))), label=dfTrn$y)
      dfValXGB = xgb.DMatrix(data=as.matrix(subset(dfVal, select=-c(y))), label=dfVal$y)
      dfTstXGB = xgb.DMatrix(data=as.matrix(subset(dfTst, select=-c(y))), label=dfTst$y)
      watchlist = list(train=dfTrnXGB, validation=dfValXGB)

      # Training using an eXtreme Gradient Boosting model
      repeat{
        # Hyperparameters
        eta = runif(1, 0.10, 0.50)
        gamma = runif(1, 0.00, 0.20)
        max_depth =isolate(MAX_DEPTH()) #round(runif(1, 3, 5), 0)
        min_child_weight = round(runif(1, 1, 6), 0)
        subsample = runif(1, 0.05, 0.25)
        colsample_bytree = runif(1, 1, 1)
        alpha = runif(1, 0, 10)
        lambda = runif(1, 1, 10)
        # Training the member of the model
        xgbTree_model = xgb.train(
          watchlist = watchlist,               # this is the dataset used for validation
          data = dfTrnXGB,                     # Training data to use
          objective = "binary:logistic",       # logistic regression for binary classification
          eval_metric = f1_score,              # Using custom f1_score as eval_metric
          maximize = TRUE,                     # Maximising the custom eval_metric
          scale_pos_weight=scale_pos_weight,   # Control the balance of positive and negative weights
          nrounds = NROUNDS,                       # Number of trees to consider
          max_depth = max_depth,               # Max Tree Depth
          min_child_weight = min_child_weight, # Min Child Tree Weight
          early_stopping_rounds = 9,          # Early stop
          eta = eta,                           # learning rate
          gamma = gamma,                       # minimum loss reduction (0 = no regularization)
          subsample = subsample,               # subsample ratio of the training instance
          colsample_bytree = colsample_bytree, # subsample ratio of columns when constructing each tree
          alpha = alpha,                       # L1 regularization term on weights
          lambda = lambda                      # L2 regularization term on weights.
        )

 

        if (as.numeric(xgb.attributes(xgbTree_model)$best_score) > 0.0) { break }

      }#end of repeat
      
      
      # ==========================================================================
      
      # Saving outputs of this training ==========================================
      # Saving the trained model to the disk
      saveRDS(xgbTree_model, paste("./",isolate(input$sTgt),"/xgb_model_classwgt_",isolate(input$sTgt),"_",str_pad(att,4,pad="0"),".rds",sep=""))

      # Saving the idx_sets
      write.table(t(as.matrix(trn_idx)), paste("./",isolate(input$sTgt),"/xgb_model_classwgt_",isolate(input$sTgt),"_dataset_idx_trn.csv",sep=""), append=T, row.names=F, 
                  col.names=!file.exists(paste("./",isolate(input$sTgt),"/xgb_model_classwgt_",isolate(input$sTgt),"_dataset_idx_trn.csv",sep="")), sep=",")
      write.table(t(as.matrix(val_idx)), paste("./",isolate(input$sTgt),"/xgb_model_classwgt_",isolate(input$sTgt),"_dataset_idx_val.csv",sep=""), append=T, row.names=F, 
                  col.names=!file.exists(paste("./",isolate(input$sTgt),"/xgb_model_classwgt_",isolate(input$sTgt),"_dataset_idx_val.csv",sep="")), sep=",")
      write.table(t(as.matrix(tst_idx)), paste("./",isolate(input$sTgt),"/xgb_model_classwgt_",isolate(input$sTgt),"_dataset_idx_tst.csv",sep=""), append=T, row.names=F, 
                  col.names=!file.exists(paste("./",isolate(input$sTgt),"/xgb_model_classwgt_",isolate(input$sTgt),"_dataset_idx_tst.csv",sep="")), sep=",")
    

      # Computing the predict results
      dfRes[[att]] = data.frame(obsv=dfDatasetTgt()$y, pred=predict(xgbTree_model, newdata=dfXGB()))
  
      dfResTrn[[att]] = data.frame(obsv=dfTrn$y, pred=predict(xgbTree_model, newdata=dfTrnXGB))
      dfResVal[[att]] = data.frame(obsv=dfVal$y, pred=predict(xgbTree_model, newdata=dfValXGB))
      dfResTst[[att]] = data.frame(obsv=dfTst$y, pred=predict(xgbTree_model, newdata=dfTstXGB))

      # Computing the error metrics
      errMetricsTrn = c(Metrics::accuracy(dfResTrn[[att]]$obsv, round(dfResTrn[[att]]$pred,0)),
                        Metrics::recall(dfResTrn[[att]]$obsv, round(dfResTrn[[att]]$pred,0)),
                        Metrics::precision(dfResTrn[[att]]$obsv, round(dfResTrn[[att]]$pred,0)), 
                        Metrics::f1(dfResTrn[[att]]$obsv, round(dfResTrn[[att]]$pred,0)))
      errMetricsVal = c(Metrics::accuracy(dfResVal[[att]]$obsv, round(dfResVal[[att]]$pred,0)),
                        Metrics::recall(dfResVal[[att]]$obsv, round(dfResVal[[att]]$pred,0)),
                        Metrics::precision(dfResVal[[att]]$obsv, round(dfResVal[[att]]$pred,0)), 
                        Metrics::f1(dfResVal[[att]]$obsv, round(dfResVal[[att]]$pred,0)))
      errMetricsTst = c(Metrics::accuracy(dfResTst[[att]]$obsv, round(dfResTst[[att]]$pred,0)),
                        Metrics::recall(dfResTst[[att]]$obsv, round(dfResTst[[att]]$pred,0)),
                        Metrics::precision(dfResTst[[att]]$obsv, round(dfResTst[[att]]$pred,0)), 
                        Metrics::f1(dfResTst[[att]]$obsv, round(dfResTst[[att]]$pred,0)))
      errMetrics[[att]] = rbind(errMetricsTrn, errMetricsVal, errMetricsTst)
      rownames(errMetrics[[att]]) = c("Training", "Validation", "Test")
      colnames(errMetrics[[att]]) = c("accuracy", "recall", "precision", "f1_score")
      errMetrics[[att]][,4] = (2*errMetrics[[att]][,3]*errMetrics[[att]][,2])/(errMetrics[[att]][,3]+errMetrics[[att]][,2]) # Fixing f1_score
      print(round(errMetrics[[att]],3))

      # Computing the variable importance
      dfVarImp[[att]] = xgb.importance(feature_names=isolate(sfeats()),model=xgbTree_model)
      dfVarImp[[att]] = dfVarImp[[att]][order(dfVarImp[[att]]$Feature),]
      
      # Computing the mean SHAP contribution to variable importance
      shapExp = xgb.plot.shap(dfSHAPsm(), model=xgbTree_model, plot=F, top_n=dim(dfSHAPsm())[2], subsample=NULL) #Arthur: subsample=1
      shapImp = colMeans(abs(shapExp$shap_contrib))
      shapImp = shapImp[order(names(shapImp))]
      dfVarImp[[att]] = cbind(dfVarImp[[att]], SHAP=shapImp)

      if(att==1){sample(1:nrow(shapExp$shap_contrib),1000)}->shap_idx
      
      # Storing the sample of SHAP results for plot_shap_beeswarm
      dfSHAP[[1]][[att]] = shapExp$shap_contrib[shap_idx,]
      dfSHAP[[2]][[att]] = as.matrix(shapExp$data[shap_idx,])
      # ==========================================================================
    
    }#end of for on: atts
    

    # Saving the list objects with the individual attempts to the disk
    saveRDS(dfRes, paste("./",isolate(input$sTgt),"/xgb_model_classwgt_",isolate(input$sTgt),"_dfRes.rds",sep=""))
    saveRDS(dfResTrn, paste("./",isolate(input$sTgt),"/xgb_model_classwgt_",isolate(input$sTgt),"_dfResTrn.rds",sep=""))
    saveRDS(dfResVal, paste("./",isolate(input$sTgt),"/xgb_model_classwgt_",isolate(input$sTgt),"_dfResVal.rds",sep=""))
    saveRDS(dfResTst, paste("./",isolate(input$sTgt),"/xgb_model_classwgt_",isolate(input$sTgt),"_dfResTst.rds",sep=""))
    saveRDS(errMetrics, paste("./",isolate(input$sTgt),"/xgb_model_classwgt_",isolate(input$sTgt),"_errMetrics.rds",sep=""))
    saveRDS(dfVarImp, paste("./",isolate(input$sTgt),"/xgb_model_classwgt_",isolate(input$sTgt),"_varImportance.rds",sep=""))
    names(dfSHAP) = c("shap_contrib", "data")
    saveRDS(dfSHAP, paste("./",isolate(input$sTgt),"/xgb_model_classwgt_",isolate(input$sTgt),"_SHAP.rds",sep=""))
  
    # Computing the mean and st_dev values of the predictions
    dfRes_atts = matrix(unlist(lapply(dfRes, subset, select="pred")), ncol=atts(), byrow=F)
    dfRes_Data = data.frame(dfDatasetTgt(), pred_mean=rowMeans(dfRes_atts), pred_sd=rowSds(dfRes_atts))
    write.csv(dfRes_Data, paste("./",isolate(input$sTgt),"/xgb_model_classwgt_",isolate(input$sTgt),"_dfRes.csv",sep=""),row.names=F)
  
    # Storing the error metrics
    errMetrics_atts = as.data.frame(apply(simplify2array(errMetrics), 1:2, mean))
    write.csv(round(errMetrics_atts,3), paste("./",isolate(input$sTgt),"/xgb_model_classwgt_",isolate(input$sTgt),"_errMetrics.csv",sep=""),row.names=F)
      
    # Storing the variable importance
    dfVarImpGain = reduce(lapply(dfVarImp, "[", , c("Feature", "Gain")),  full_join, by="Feature")
    dfVarImpCovr = reduce(lapply(dfVarImp, "[", , c("Feature", "Cover")), full_join, by="Feature")
    dfVarImpFreq = reduce(lapply(dfVarImp, "[", , c("Feature", "Frequency")), full_join, by="Feature")
    dfVarImpSHAP = reduce(lapply(dfVarImp, "[", , c("Feature", "SHAP")), full_join, by="Feature")
    dfVarImpGain = data.frame(dfVarImpGain[,1], "Gain"=rowMeans(dfVarImpGain[,2:NCOL(dfVarImpGain)], na.rm=T))
    dfVarImpCovr = data.frame(dfVarImpCovr[,1], "Cover"=rowMeans(dfVarImpCovr[,2:NCOL(dfVarImpCovr)], na.rm=T))
    dfVarImpFreq = data.frame(dfVarImpFreq[,1], "Frequency"=rowMeans(dfVarImpFreq[,2:NCOL(dfVarImpFreq)], na.rm=T))
    dfVarImpSHAP = data.frame(dfVarImpSHAP[,1], "SHAP"=rowMeans(dfVarImpSHAP[,2:NCOL(dfVarImpSHAP)], na.rm=T))
    dfVarImp_atts = data.frame(Feature=dfVarImpGain$Feature, 
                               Gain=dfVarImpGain$Gain, 
                               Cover=dfVarImpCovr$Cover, 
                               Frequency=dfVarImpFreq$Frequency, 
                               SHAP=dfVarImpSHAP$SHAP)
    dfVarImp_atts = dfVarImp_atts[order(dfVarImp_atts$Feature),]
    write.csv(dfVarImp_atts, paste("./",isolate(input$sTgt),"/xgb_model_classwgt_",isolate(input$sTgt),"_variableImportance.csv",sep=""),row.names=F)

    file_ready_signal(TRUE)

    
  },ignoreNULL=TRUE)
  
  
  observe({
    
    req(file_ready_signal())

    print(paste0("Processing target variable: ", isolate(input$sTgt)))

    # Getting the features
    sFeats =isolate(sfeats())  #readRDS(list.files(paste("./",sTgt,sep=""), ".rds", full.names=T)[1])$feature_names

    # Creating the database to extract the data and results
    iCols = which(names(dfData_pred()) %in% c(sIDs, sFeats, isolate(input$sTgt)))
    dfDataset = dfData_pred()[,iCols] 
    dfRes = dfDataset[,sIDs]
    
    # dfDataset = na.omit(dfDataset)
    dfDataset[is.na(dfDataset)] = 0
    iCols = which(names(dfDataset) %in% c(sIDs))
    dfIDs = dfDataset[,iCols] 
    
    # Preparing the data for a xgb model
    iCols = which(!(names(dfDataset) %in% sIDs))
    dfXGB = xgb.DMatrix(data=as.matrix(dfDataset[,iCols]))
    
    # Computing the mean and st_dev values of the predictions
    sModels = list.files(paste("./",isolate(input$sTgt),sep=""), ".rds", full.names=T)

    sModels = sModels[1:atts()]
    lModels = split(sModels, ceiling(seq_along(sModels)/100))
    fPredict = function(sModel) { predict(readRDS(sModel), newdata=dfXGB) }
    dfDatasetRes = NULL
    pbi = 0; pb = txtProgressBar(min=0, max=length(lModels), initial=0, style=3)
    for (sModels in lModels) {
      mDatasetRes = lapply(sModels, fPredict)
      mDatasetRes = matrix(unlist(mDatasetRes), nrow=NROW(dfIDs), ncol=length(sModels), byrow=F)
      dfDatasetRes = cbind(dfDatasetRes, mDatasetRes)
      rm(mDatasetRes); gc()
      pbi = pbi+1; setTxtProgressBar(pb,pbi)
    }
    dfDatasetRes = data.frame(pred_mean=rowMeans(dfDatasetRes), 
                              pred_sd=rowSds(dfDatasetRes), 
                              count_50=rowMeans(dfDatasetRes>0.50), 
                              count_90=rowMeans(dfDatasetRes>0.90),
                              count_95=rowMeans(dfDatasetRes>0.95),
                              count_99=rowMeans(dfDatasetRes>0.99)); gc()
    
    # Consolidating the results
    dfDatasetRes = cbind(dfDataset[,sIDs], dfDatasetRes)
    names(dfDatasetRes)[NCOL(dfDatasetRes)-5] = paste("pred_",isolate(input$sTgt),sep="")
    names(dfDatasetRes)[NCOL(dfDatasetRes)-4] = paste("sd_",isolate(input$sTgt),sep="")
    names(dfDatasetRes)[NCOL(dfDatasetRes)-3] = paste("cnt50_",isolate(input$sTgt),sep="")
    names(dfDatasetRes)[NCOL(dfDatasetRes)-2] = paste("cnt90_",isolate(input$sTgt),sep="")
    names(dfDatasetRes)[NCOL(dfDatasetRes)-1] = paste("cnt95_",isolate(input$sTgt),sep="")
    names(dfDatasetRes)[NCOL(dfDatasetRes)] = paste("cnt99_",isolate(input$sTgt),sep="")
    
    # Consolidating the results
    dfRes = merge(dfRes, dfDatasetRes, by=sIDs, all.x=T)
    
    # Storing the predictions of the model to the disk
    write.csv(dfRes, paste("./xgb_model_predictions_",isolate(input$sTgt),".csv",sep=""), row.names=F)

    file_ready_signal_pred(TRUE)
    
  })
  
  
  
  observe({
    
    req(file_ready_signal_pred())
     
    nrounds = NROUNDS
    dfRes = list()
    sModels = list.files(paste("./",isolate(isolate(input$sTgt)),sep=""), ".rds", full.names=T)
    sModels = sModels[1:isolate(atts())]
    
    xgbModel = lapply(sModels, readRDS)
    #xgbModel = lapply(xgbModel, "attr", "evaluation_log")
    xgbModel = lapply(xgbModel, "[[", "evaluation_log")
    
    for (i in 1:length(xgbModel)) {
      # Checking if first results are zero

      repeat{

        if (xgbModel[[i]][1,2] == 0) { 
          xgbModel[[i]] = xgbModel[[i]][-1,] 
        } else { break }
      }

      # Combining the data
      nrows = NROW(xgbModel[[i]])
      dfAdd = data.frame(iter=seq(1:nrounds), train_f1_score=NA, validation_f1_score=NA)
      if (nrows > nrounds) {
        dfAdd[1:nrounds,] = xgbModel[[i]][1:nrounds,]
      } else {
        dfAdd[1:nrows,] = xgbModel[[i]][1:nrows,]
      }
      dfAdd = rbind(c(0,0,0), dfAdd)
      dfAdd$iter = seq(1:NROW(dfAdd))
      dfAdd$att = i
      dfRes[[i]] = dfAdd
    }
    
    file_ready_signal(NULL)
    file_ready_signal_pred(NULL)
    
    dfRes_finale(as.data.frame(do.call(rbind, dfRes)))
    
    w$hide()
    
  })

  observe({
    
    req(dfRes_finale())
    
    tmp<-read.csv(paste("./",isolate(input$sTgt),"/xgb_model_classwgt_",isolate(input$sTgt),"_variableImportance.csv",sep=""))
    tmp<- predict(preProcess(tmp, method = "range"), tmp)
    dfVarImpNorm(tmp[order(tmp$SHAP, decreasing=T),])

  })
  
  observe({
    req(dfRes_finale())
    dfSHAP_react(readRDS(paste("./",isolate(input$sTgt),"/xgb_model_classwgt_",isolate(input$sTgt),"_SHAP.rds",sep="")))
  })
  

  renderPlot({
    
    req(dfRes_finale())
    
    if(input$metrics=="F1Score"){
      
      # Saving the figure of the evaluation_log_train
      ggplot(dfRes_finale()) +   
        geom_line(aes(x=iter , y=validation_f1_score, group=att), alpha=0.5, colour="red", linewidth=0.6) +
        geom_line(aes(x=iter , y=train_f1_score, group=att), alpha=0.5, colour="blue", linewidth=0.6) +
        scale_y_continuous(limits = c(0, 1), expand=c(0,0)) + 
        scale_x_continuous(limits = c(0, NROUNDS), expand=c(0,0)) +         
        theme_bw() + 
        xlab("Number of Decision Trees") + 
        ylab("F1-Score") + 
        theme(text=element_text(size=21),
              strip.text.x = element_blank(),
              strip.background = element_blank(),
              legend.title=element_blank(), 
              legend.position=c(.88,.16),
              plot.margin=margin(20,20,2,2))->grafico      
      
    }
    
    if(input$metrics=="variableImportance"){

    
      ggplot(melt(dfVarImpNorm()), aes(reorder(Feature, +value), value, group=variable, fill=variable)) +
        geom_bar(stat = 'identity', position = position_dodge(0.8))+
        theme_bw() + 
        coord_flip() + 
        scale_fill_manual(values = c("#e26d5c","#ffe1a8", "#c9cba3", "lightblue")) + 
        scale_y_continuous(limits = c(0, 1), expand=c(0,0)) +
        xlab("") + 
        ylab("") + 
        theme(text=element_text(size=21),
              strip.text.x = element_blank(),
              strip.background = element_blank(),
              legend.title=element_blank(), 
              legend.position.inside=c(.88,.16),
              plot.margin=margin(20,20,2,2))->grafico
       
    }  
    
    if(input$metrics=="SHAP"){
      
      plot_shap_beeswarm(dfSHAP_react()[[1]][[1]], dfSHAP_react()[[2]][[1]], size=1.0, cex=0.15, sort=T,
                         corral="random", scheme=c("cornflowerblue",alpha("white",0.01),"darkred"))+
        theme(text=element_text(size=21),
              strip.text.x = element_blank(),
              strip.background = element_blank(),
              legend.title=element_blank(), 
              legend.position.inside=c(.88,.16),
              plot.margin=margin(20,50,2,2),
              panel.grid=element_blank(),
              panel.grid.major.y=element_line(colour="lightgray",linewidth=0.5),
              axis.line.y=element_blank(),
              axis.ticks.y=element_blank())->grafico
      
    }      
    
    
    
    return(grafico)
    
  })->output$pVarImp  
  
  
  
  renderPlot({
    
    req(dfRes_finale())
    
    w$show()
    sColPal = sCols[[which(isolate(input$sTgt) == sVars)]]
    sColPal = c(rep("white", 9), sColPal)
    
    dfData = read.csv(paste("./xgb_model_predictions_",isolate(input$sTgt),".csv",sep=""))
    dfData[isolate(input$sTgt)] = dfData[paste("pred_",isolate(input$sTgt),sep="")]
    dfCSV = dfData[,names(dfData) %in% c("X", "Y", "Time", isolate(input$sTgt))]
    sfCSV = st_as_sf(dfCSV, coords = c("X", "Y"), crs=st_crs(sfGrid))
    sfCSV = st_transform(sfCSV, "EPSG:3035")
    sfExt = st_bbox(sfCSV)
    sfBack = st_transform(world, st_crs(sfCSV))
    sfWater = st_difference(st_make_valid(st_as_sfc(sfExt)), st_make_valid(st_union(st_crop(sfBack, sfExt))))
    ggplot() +
      geom_sf(data=sfBack, fill="grey80", linewidth=0.3, alpha=0.5) +
      geom_sf(data=sfCSV, aes_string(colour=isolate(input$sTgt)), shape=15, size=1.50, linewidth=0.3, alpha=1.0) +
      geom_sf(data=sfWater, fill="white", linewidth=0.3, alpha=1.0) +
      geom_sf(data=sfBack, fill="white", linewidth=0.3, alpha=0.0) +
      coord_sf(crs=st_crs(sfBack), xlim=c(sfExt$xmin,sfExt$xmax), ylim=c(sfExt$ymin,sfExt$ymax), expand=F) +
      scale_colour_gradientn(colours=sColPal, breaks=round(sBrk,2), limits=limits, oob=scales::squish,guide = guide_colorbar(frame.col = "black")) +
      labs(title=isolate(input$sTgt), y=NULL, x=NULL, colour=NULL) + # , colour=sVar) +
      theme_bw() + 
      theme(plot.title = element_text(size = 24, face = "bold"), 
            panel.grid.major = element_line(color = gray(.5), linetype = "dashed", linewidth = 0.1), 
            panel.background = element_rect(fill = "white"), 
            legend.text = element_text(size = 10),
            legend.key.height = unit(3.5, 'cm'),
            legend.key.width = unit(1, 'cm'),
            legend.key=element_rect(colour="black"),
            legend.position = "right")->grafico
    
    w$hide()  
    return(grafico)
    
  })->output$map
  
  
  reactive({
    
    req(dfRes_finale())
    
    # Getting the main drivers
    tmp= read.csv(paste("./",isolate(input$sTgt),"/xgb_model_classwgt_",isolate(input$sTgt),"_variableImportance.csv",sep=""))
    tmp[order(tmp$SHAP, decreasing=T),]
    
  })->df_var_imp  
    
    
      
    
  renderUI({
  
    #Sorted Features
    req(df_var_imp()$Feature)
    
    selectInput("selected_map_feature","Select a feature to visualize:",choices =df_var_imp()$Feature,selected=df_var_imp()$Feature[1])
    
    
  })->output$feature_selector
  
  
  renderPlot({
    
    req(input$selected_map_feature)
    req(dfRes_finale())
    
    w$show()
    
    palSpectral = rev(brewer.pal(9,"Spectral")); palSpectral[5] = "white"
    sColPals = list(palSpectral, palSpectral, rev(brewer.pal(9,"RdBu")), brewer.pal(9,"YlOrRd"), rev(brewer.pal(9,"GnBu")), brewer.pal(9,"BuPu"), palSpectral, 
                    rev(brewer.pal(9,"PuOr")), rev(brewer.pal(9,"PuOr")), rev(brewer.pal(9,"PuOr")), rev(brewer.pal(9,"PuOr")), rev(brewer.pal(9,"PuOr")), brewer.pal(9,"PuOr"), rev(brewer.pal(9,"PuOr")), 
                    brewer.pal(9,"RdBu"), brewer.pal(9,"RdBu"), brewer.pal(9,"RdBu"), brewer.pal(9,"RdBu"), brewer.pal(9,"RdBu"), brewer.pal(9,"BrBG"), brewer.pal(9,"PiYG"), 
                    c("grey95",brewer.pal(9,"YlOrRd")), c("grey95",brewer.pal(9,"YlOrRd")), c("grey95",brewer.pal(9,"GnBu")), c("grey95",brewer.pal(9,"GnBu")), c("grey95",brewer.pal(9,"GnBu")), c("grey95",brewer.pal(9,"BuPu")), c("grey95",brewer.pal(9,"BuPu")))
    brk_anom = c(-3.0,-2.0,-1.5,-1.0,-0.50,0.50,1.0,1.5,2.0,3.0)
    brk_seq = seq(0,1,0.1)
    sBrks = list(seq(100000,102500,length.out=10),seq(83500,102000,length.out=10),seq(265,300,length.out=10),seq(270,305,length.out=10),seq(260,295,length.out=10),seq(0,9,length.out=10),seq(51000,57000,length.out=10),
                 brk_anom*300, brk_anom*1000, brk_anom*1, brk_anom*1, brk_anom*1, brk_anom*1, brk_anom*300,
                 brk_anom, brk_anom, brk_anom, brk_anom, brk_anom, brk_anom, brk_anom,
                 brk_seq,brk_seq,brk_seq,brk_seq,brk_seq,brk_seq,brk_seq)
    names(sColPals) = sVarImps
    names(sBrks) = sVarImps

    which(input$selected_map_feature==df_var_imp()$Feature)->i
    
     # 1 means most important
    sVarImp = df_var_imp()$Feature[i]
    sColPal = sColPals[[which(sVarImps == sVarImp)]]
    sBrk = sBrks[[which(sVarImps == sVarImp)]]
    
    # Processing the data
    dfData<-dfData_pred() 
    dfData$Time = paste(dfData$Year, str_pad(dfData$Month, 2, pad="0"), "01", sep="-")
    dfCSV = dfData[,names(dfData) %in% c("X", "Y", "Time", sVarImp)]
    sfCSV = st_as_sf(dfCSV, coords = c("X", "Y"), crs=st_crs(sfGrid))
    sfCSV = st_transform(sfCSV, "EPSG:3035")
    sfExt = st_bbox(sfCSV)
    sfBack = st_transform(world, st_crs(sfCSV))
    sfWater = st_difference(st_make_valid(st_as_sfc(sfExt)), st_make_valid(st_union(st_crop(sfBack, sfExt))))
    limits = c(min(sBrk),max(sBrk))
    
    ggplot() +
      geom_sf(data=sfBack, fill="grey80", linewidth=0.3, alpha=0.5) +
      geom_sf(data=sfCSV, aes_string(colour=sVarImp), shape=15, size=1.50, linewidth=0.3, alpha=1.0) +
      geom_sf(data=sfWater, fill="white", linewidth=0.3, alpha=1.0) +
      geom_sf(data=sfBack, fill="white", linewidth=0.3, alpha=0.0) +
      coord_sf(crs=st_crs(sfBack), xlim=c(sfExt$xmin,sfExt$xmax), ylim=c(sfExt$ymin,sfExt$ymax), expand=F) +
      scale_colour_gradientn(colours=sColPal, breaks=round(sBrk,1), limits=limits, oob=scales::squish,guide = guide_colorbar(frame.col = "black")) +
      labs(title=sVarImp,subtitle=paste0("Feature Importance: ",i," Number of features analysed: ",nrow(df_var_imp())), y=NULL, x=NULL, colour=NULL) + # , colour=sVar) +
      theme_bw() + 
      theme(plot.title = element_text(size = 24, face = "bold"), 
            panel.grid.major = element_line(color = gray(.5), linetype = "dashed", linewidth = 0.1), 
            panel.background = element_rect(fill = "white"), 
            legend.text = element_text(size = 10),
            legend.key.height = unit(3.5, 'cm'),
            legend.key.width = unit(1, 'cm'),
            legend.key=element_rect(colour="black"),
            legend.position = "right")->grafico
    
    w$hide()
    return(grafico)

    
  })->output$feature
  
  

  
}