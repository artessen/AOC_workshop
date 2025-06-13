# Loading the required libraries ===============================================
library(caret)
library(xgboost)
library(ggplot2)
library(lattice)
library(stringr)
library(matrixStats)
library(tidyverse)
# ==============================================================================

# Defining the custom objective and metric functions ===========================
source("00.a.aux_AuxiliaryFunctions.R")
# ==============================================================================

# Reading the input data =======================================================
# Reading the database (example as a sample database)
#dfData = as.data.frame(fread("dfDatabase_sample.csv"))
dfData = as.data.frame(read.csv("dfDatabase_sample.csv"))

# Defining the ID and Target columns
sIDs = c("X","Y","Time","Month","Year","Day")
sTgts = c("ColdS", "Drought", "HeatW", "HotDry", "RainD", "RainS", "TSumS", "TSumD")

# Defining the possible Feature columns
sERA5mean = c("msl", "sp", "t2m", "tmax", "tmin", "tp", "z500")
sERA5anom = c("msl", "sp", "t2m", "tmax", "tmin", "tp", "z500")
sERA5spec = c("daystmax25a30C", "daystmax30C", 
              "daystmin0C", "daystminm10C", "daystminm18C", 
              "daystp01mm", "daystp30mm")
sSPI = c("spi01","spi03","spi06","spi09","spi12")
sCEMS = c("smant", "fpanv")
sERA5mean = paste(sERA5mean, "_mean", sep="")
sERA5anom = paste(sERA5anom, "_anom", sep="")

# Defining the list of Feature columns as a function of the Target variable
lFeats = list()
lFeats[[1]] = c(sERA5mean, sERA5anom, sERA5spec, sSPI, sCEMS) # ColdS
lFeats[[2]] = c(sERA5mean, sERA5anom, sERA5spec, sSPI, sCEMS) # Drought
lFeats[[3]] = c(sERA5mean, sERA5anom, sERA5spec, sSPI, sCEMS) # HeatW
lFeats[[4]] = c(sERA5mean, sERA5anom, sERA5spec, sSPI, sCEMS) # HotDry
lFeats[[5]] = c(sERA5mean, sERA5anom, sERA5spec, sSPI, sCEMS) # RainD
lFeats[[6]] = c(sERA5mean, sERA5anom, sERA5spec, sSPI, sCEMS) # RainS
lFeats[[7]] = c(sERA5mean, sERA5anom, sERA5spec, sSPI, sCEMS) # TSumS
lFeats[[8]] = c(sERA5mean, sERA5anom, sERA5spec, sSPI, sCEMS) # TSumD

# Getting dates where AOC exist
dfDates = sort(unique(dfData$Time[rowSums(dfData[names(dfData) %in% sTgts], na.rm=T) > 0]))

# Defining the number of training attempts
atts = 3
# ==============================================================================

# For every target variable ====================================================
# Training the model for the selected target variable
for (i in 1:length(sTgts)) {
  
  # Progress message
  sTgt = sTgts[i]
  sFeats = lFeats[[i]]
  message("Processing target variable: ", sTgt)
  
  # Creating output folder directory
  if (dir.exists(sTgt)) { unlink(sTgt, recursive=TRUE) }
  dir.create(sTgt)
  
  # Creating the training database
  dfDataset = dfData[,names(dfData) %in% c(sIDs, sFeats, sTgts),drop=FALSE] 
  dfDataset = na.omit(dfDataset)
  # Keeping only dates where the AOC exist
  dfDataset = dfDataset[dfDataset$Time %in% dfDates,]
  idT = which(dfDataset[[sTgt]]==1) 
  idT_dates = sort(unique(dfDataset[idT,]$Time))
  dfDataset = dfDataset[dfDataset$Time %in% idT_dates,]
  # Shuffling the data and getting the IDs
  dfDataset = dfDataset[sample(NROW(dfDataset)),]
  dfIDs = dfDataset[,names(dfDataset) %in% c(sIDs)] 
  # Preparing the dfDatasetTgt object
  dfDatasetTgt = dfDataset[,names(dfDataset) %in% c(sFeats,sTgt),drop=FALSE]
  names(dfDatasetTgt)[names(dfDatasetTgt)==sTgt] = "y"
  # Saving the dfDataset of this target variable
  write.csv(dfDatasetTgt, paste("./",sTgt,"/xgb_model_classwgt_",sTgt,"_dataset.csv",sep=""), row.names=F)
  
  # Preparing the data for a xgb model
  dfXGB = xgb.DMatrix(data=as.matrix(subset(dfDatasetTgt, select=-c(y))), label=dfDatasetTgt$y)
  dfSHAPsm = Matrix::Matrix(as.matrix(subset(dfDatasetTgt, select=-c(y))), sparse=T)
  
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
  shap_idx = sample(1:dim(dfDataset)[1], 1000)
  for (att in 1:atts) {
    # Training block of the model ==============================================
    message("  Processing fold ", att, " out of ", atts)
    # Specifying the % train-validation-test split
    trn_idx = as.vector(createDataPartition(dfDataset[[1]][sample(NROW(dfDataset))], p=0.50, list=FALSE))
    tst_idx = seq(1,NROW(dfDataset),1)[-trn_idx]
    # Specifying the % train-validation-test split
    trn_idx = seq(1,NROW(dfDataset),1)[-tst_idx]
    val_idx = createDataPartition(trn_idx, p=0.25, list=FALSE); trn_idx = trn_idx[-val_idx]
    # Creating the training, validation and test sets
    dfTrn = dfDatasetTgt[trn_idx,,drop=FALSE]
    dfVal = dfDatasetTgt[val_idx,,drop=FALSE]
    dfTst = dfDatasetTgt[tst_idx,,drop=FALSE]
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
    repeat {
      # Hyperparameters
      eta = runif(1, 0.10, 0.50)
      gamma = runif(1, 0.00, 0.20)
      max_depth = round(runif(1, 3, 5), 0)
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
        nrounds = 50,                       # Number of trees to consider
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
      if (xgbTree_model$best_score > 0.0) { break }
    }
    # ==========================================================================
    
    # Saving outputs of this training ==========================================
    # Saving the trained model to the disk
    saveRDS(xgbTree_model, paste("./",sTgt,"/xgb_model_classwgt_",sTgt,"_",str_pad(att,4,pad="0"),".rds",sep=""))
    
    # Saving the idx_sets
    write.table(t(as.matrix(trn_idx)), paste("./",sTgt,"/xgb_model_classwgt_",sTgt,"_dataset_idx_trn.csv",sep=""), append=T, row.names=F, 
                col.names=!file.exists(paste("./",sTgt,"/xgb_model_classwgt_",sTgt,"_dataset_idx_trn.csv",sep="")), sep=",")
    write.table(t(as.matrix(val_idx)), paste("./",sTgt,"/xgb_model_classwgt_",sTgt,"_dataset_idx_val.csv",sep=""), append=T, row.names=F, 
                col.names=!file.exists(paste("./",sTgt,"/xgb_model_classwgt_",sTgt,"_dataset_idx_val.csv",sep="")), sep=",")
    write.table(t(as.matrix(tst_idx)), paste("./",sTgt,"/xgb_model_classwgt_",sTgt,"_dataset_idx_tst.csv",sep=""), append=T, row.names=F, 
                col.names=!file.exists(paste("./",sTgt,"/xgb_model_classwgt_",sTgt,"_dataset_idx_tst.csv",sep="")), sep=",")
    
    # Computing the predict results
    dfRes[[att]] = data.frame(obsv=dfDatasetTgt$y, pred=predict(xgbTree_model, newdata=dfXGB))
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
    dfVarImp[[att]] = xgb.importance(model=xgbTree_model)
    dfVarImp[[att]] = dfVarImp[[att]][order(dfVarImp[[att]]$Feature),]
    
    # Computing the mean SHAP contribution to variable importance
    shapExp = xgb.plot.shap(dfSHAPsm, model=xgbTree_model, plot=F, top_n=dim(dfSHAPsm)[2], subsample=1)
    shapImp = colMeans(abs(shapExp$shap_contrib))
    shapImp = shapImp[order(names(shapImp))]
    dfVarImp[[att]] = cbind(dfVarImp[[att]], SHAP=shapImp)
    
    # Storing the sample of SHAP results for plot_shap_beeswarm
    dfSHAP[[1]][[att]] = shapExp$shap_contrib[shap_idx,]
    dfSHAP[[2]][[att]] = as.matrix(shapExp$data[shap_idx,])
    # ==========================================================================
    
  }
  
  # Saving the list objects with the individual attempts to the disk
  saveRDS(dfRes, paste("./",sTgt,"/xgb_model_classwgt_",sTgt,"_dfRes.rds",sep=""))
  saveRDS(dfResTrn, paste("./",sTgt,"/xgb_model_classwgt_",sTgt,"_dfResTrn.rds",sep=""))
  saveRDS(dfResVal, paste("./",sTgt,"/xgb_model_classwgt_",sTgt,"_dfResVal.rds",sep=""))
  saveRDS(dfResTst, paste("./",sTgt,"/xgb_model_classwgt_",sTgt,"_dfResTst.rds",sep=""))
  saveRDS(errMetrics, paste("./",sTgt,"/xgb_model_classwgt_",sTgt,"_errMetrics.rds",sep=""))
  saveRDS(dfVarImp, paste("./",sTgt,"/xgb_model_classwgt_",sTgt,"_varImportance.rds",sep=""))
  names(dfSHAP) = c("shap_contrib", "data")
  saveRDS(dfSHAP, paste("./",sTgt,"/xgb_model_classwgt_",sTgt,"_SHAP.rds",sep=""))
  
  # Computing the mean and st_dev values of the predictions
  dfRes_atts = matrix(unlist(lapply(dfRes, subset, select="pred")), ncol=atts, byrow=F)
  dfRes_Data = data.frame(dfDatasetTgt, pred_mean=rowMeans(dfRes_atts), pred_sd=rowSds(dfRes_atts))
  write.csv(dfRes_Data, paste("./",sTgt,"/xgb_model_classwgt_",sTgt,"_dfRes.csv",sep=""),row.names=F)
  
  # Storing the error metrics
  errMetrics_atts = as.data.frame(apply(simplify2array(errMetrics), 1:2, mean))
  write.csv(round(errMetrics_atts,3), paste("./",sTgt,"/xgb_model_classwgt_",sTgt,"_errMetrics.csv",sep=""),row.names=F)
  
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
  write.csv(dfVarImp_atts, paste("./",sTgt,"/xgb_model_classwgt_",sTgt,"_variableImportance.csv",sep=""),row.names=F)
}
# ==============================================================================
