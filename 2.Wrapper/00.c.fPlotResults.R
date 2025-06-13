# Function to predict an XGBoost model for detecting AOCs
fPlotResults = function(Target, TrainingAttempts=3) {
  
  # Loading the required libraries ===============================================
  library(RColorBrewer)
  library(sf)
  library(rnaturalearth)
  library(ggplot2)
  library(nestedcv)
  library(caret)
  library(reshape2)
  library(stringr)
  # ==============================================================================
  
  # Plotting the training evolution results ======================================
  sAOC = Target
  atts = TrainingAttempts
  nrounds = 50
  dfRes = list()
  sModels = list.files(paste("./",sAOC,sep=""), ".rds", full.names=T)
  sModels = sModels[1:atts]
  xgbModel = lapply(sModels, readRDS)
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
  dfRes = as.data.frame(do.call(rbind, dfRes))
  # Saving the figure of the evaluation_log_train
  pVarImp = ggplot(dfRes) +   
    geom_line(aes(x=iter , y=validation_f1_score, group=att), alpha=0.5, colour="red", linewidth=0.6) +
    geom_line(aes(x=iter , y=train_f1_score, group=att), alpha=0.5, colour="blue", linewidth=0.6) +
    scale_y_continuous(limits = c(0, 1), oob=scales::squish) + 
    theme_bw() + 
    xlab("Number of trees") + 
    ylab("F1-Score") + 
    theme(text=element_text(size=21),
          strip.text.x = element_blank(),
          strip.background = element_blank(),
          legend.title=element_blank(), 
          legend.position=c(.88,.16))
  plot(pVarImp)
  # ==============================================================================
  
  # Plotting the variable importance of a single AOC =============================
  dfVarImp = read.csv(paste("./",sAOC,"/xgb_model_classwgt_",sAOC,"_variableImportance.csv",sep=""))
  dfVarImpNorm = predict(preProcess(dfVarImp, method = "range"), dfVarImp)
  dfVarImpNorm = dfVarImpNorm[order(dfVarImpNorm$SHAP, decreasing=T),]
  pVarImp = ggplot(melt(dfVarImpNorm), aes(reorder(Feature, +value), value, group=variable, fill=variable)) +
    geom_bar(stat = 'identity', position = position_dodge(0.8))+
    theme_bw() + 
    coord_flip() + 
    scale_fill_manual(values = c("#e26d5c","#ffe1a8", "#c9cba3", "lightblue")) + 
    xlab("") + 
    ylab("") + 
    theme(text=element_text(size=21),
          strip.text.x = element_blank(),
          strip.background = element_blank(),
          legend.title=element_blank(), 
          legend.position.inside=c(.88,.16))
  plot(pVarImp)
  # ==============================================================================
  
  # Plotting the SHAP results of a single AOC ====================================
  dfSHAP = readRDS(paste("./",sAOC,"/xgb_model_classwgt_",sAOC,"_SHAP.rds",sep=""))
  sModels = list.files(paste("./",sAOC,sep=""), ".rds", full.names=T)
  sModels = sModels[1:atts]
  xgbModel = lapply(sModels, readRDS)
  xgbModel = lapply(xgbModel, "[[", "best_score")
  iBest = which.max(unlist(xgbModel))
  pltSHAP = plot_shap_beeswarm(dfSHAP[[1]][[iBest]], dfSHAP[[2]][[iBest]], size=1.0, cex=0.15, sort=T,
                               corral="random", scheme=c("cornflowerblue",alpha("white",0.01),"darkred"))
  plot(pltSHAP)
  # ==============================================================================
  
  # Showing the map of the predictions ===========================================
  sfGrid = read_sf("AOC_grid_clean_4326.gpkg")
  world = ne_countries(scale = "medium", returnclass = "sf")$geometry
  sVars = c("ColdS", "HeatW", "Drought", "HotDry", "RainD", "RainS", "TSumD", "TSumS")
  sNames = c("Cold spell", "Heat wave", "Drought", "Hot and dry conditions", 
             "Rain deficit", "Rain excess", 
             "Temperature accumulation deficit", "Temperature accumulation surplus")
  sCols = list(brewer.pal(9,"BuPu"), brewer.pal(9,"Reds"), brewer.pal(9,"Oranges"), brewer.pal(9,"OrRd"), 
               brewer.pal(9,"YlOrRd"), brewer.pal(9,"Blues"), brewer.pal(9,"GnBu"), brewer.pal(9,"YlOrBr"))
  sColPal = sCols[[which(sAOC == sVars)]]
  sColPal = c(rep("white", 9), sColPal)
  sBrk = c(0, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0)
  limits = c(0,1)
  dfData = read.csv(paste("./",sAOC,"/xgb_model_predictions_",sAOC,".csv",sep=""))
  dfData[sAOC] = dfData[paste("pred_",sAOC,sep="")]
  dfCSV = dfData[,names(dfData) %in% c("X", "Y", "Time", sAOC)]
  sfCSV = st_as_sf(dfCSV, coords = c("X", "Y"), crs=st_crs(sfGrid))
  sfCSV = st_transform(sfCSV, "EPSG:3035")
  sfExt = st_bbox(sfCSV)
  sfBack = st_transform(world, st_crs(sfCSV))
  sfWater = st_difference(st_make_valid(st_as_sfc(sfExt)), st_make_valid(st_union(st_crop(sfBack, sfExt))))
  pltAOC = ggplot() +
    geom_sf(data=sfBack, fill="grey80", linewidth=0.3, alpha=0.5) +
    geom_sf(data=sfCSV, aes_string(colour=sAOC), shape=15, size=1.50, linewidth=0.3, alpha=1.0) +
    geom_sf(data=sfWater, fill="white", linewidth=0.3, alpha=1.0) +
    geom_sf(data=sfBack, fill="white", linewidth=0.3, alpha=0.0) +
    coord_sf(crs=st_crs(sfBack), xlim=c(sfExt$xmin,sfExt$xmax), ylim=c(sfExt$ymin,sfExt$ymax), expand=F) +
    scale_colour_gradientn(colours=sColPal, breaks=round(sBrk,2), limits=limits, oob=scales::squish) +
    labs(title=sAOC, y=NULL, x=NULL, colour=NULL) + # , colour=sVar) +
    theme_bw() + 
    theme(plot.title = element_text(size = 14, face = "bold"), 
          panel.grid.major = element_line(color = gray(.5), linetype = "dashed", linewidth = 0.1), 
          panel.background = element_rect(fill = "white"), 
          legend.text = element_text(size = 10),
          legend.key.height = unit(2.5, 'cm'),
          legend.key.width = unit(0.4, 'cm'),
          legend.position = "right")
  plot(pltAOC)
  # ==============================================================================
  
  # Showing the map of selected driver ===========================================
  sfGrid = read_sf("AOC_grid_clean_4326.gpkg")
  world = ne_countries(scale = "medium", returnclass = "sf")$geometry
  sVarImps = c("msl_mean", "sp_mean", "t2m_mean", "tmax_mean", "tmin_mean", "tp_mean", "z500_mean", 
               "msl_anom", "sp_anom", "t2m_anom", "tmax_anom", "tmin_anom", "tp_anom", "z500_anom", 
               "spi01", "spi03", "spi06", "spi09", "spi12", "smant", "fpanv", 
               "daystmax25a30C", "daystmax30C", "daystmin0C", "daystminm10C", "daystminm18C", "daystp01mm", "daystp30mm")
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
  dfData = read.csv("dfDatabaseProd_sample.csv")
  dfData$Time = paste(dfData$Year, str_pad(dfData$Month, 2, pad="0"), "01", sep="-")
  # Getting the main drivers
  dfVarImp = read.csv(paste("./",sAOC,"/xgb_model_classwgt_",sAOC,"_variableImportance.csv",sep=""))
  dfVarImp = dfVarImp[order(dfVarImp$SHAP, decreasing=T),]
  # Selecting the ranking of the variable SHAP importance
  i = 1 # 1 means most important
  for (i in 1:4) {
    sVarImp = dfVarImp$Feature[i]
    sColPal = sColPals[[which(sVarImps == sVarImp)]]
    sBrk = sBrks[[which(sVarImps == sVarImp)]]
    # Processing the data
    dfCSV = dfData[,names(dfData) %in% c("X", "Y", "Time", sVarImp)]
    sfCSV = st_as_sf(dfCSV, coords = c("X", "Y"), crs=st_crs(sfGrid))
    sfCSV = st_transform(sfCSV, "EPSG:3035")
    sfExt = st_bbox(sfCSV)
    sfBack = st_transform(world, st_crs(sfCSV))
    sfWater = st_difference(st_make_valid(st_as_sfc(sfExt)), st_make_valid(st_union(st_crop(sfBack, sfExt))))
    limits = c(min(sBrk),max(sBrk))
    pltDriver = ggplot() +
      geom_sf(data=sfBack, fill="grey80", linewidth=0.3, alpha=0.5) +
      geom_sf(data=sfCSV, aes_string(colour=sVarImp), shape=15, size=1.50, linewidth=0.3, alpha=1.0) +
      geom_sf(data=sfWater, fill="white", linewidth=0.3, alpha=1.0) +
      geom_sf(data=sfBack, fill="white", linewidth=0.3, alpha=0.0) +
      coord_sf(crs=st_crs(sfBack), xlim=c(sfExt$xmin,sfExt$xmax), ylim=c(sfExt$ymin,sfExt$ymax), expand=F) +
      scale_colour_gradientn(colours=sColPal, breaks=round(sBrk,1), limits=limits, oob=scales::squish) +
      labs(title=paste(sVarImp," - Driver number ",i,sep=""), y=NULL, x=NULL, colour=NULL) + # , colour=sVar) +
      theme_bw() + 
      theme(plot.title = element_text(size = 14, face = "bold"), 
            panel.grid.major = element_line(color = gray(.5), linetype = "dashed", linewidth = 0.1), 
            panel.background = element_rect(fill = "white"), 
            legend.text = element_text(size = 10),
            legend.key.height = unit(2.5, 'cm'),
            legend.key.width = unit(0.4, 'cm'),
            legend.position = "right")
    plot(pltDriver)
  }
  # ==============================================================================
}
