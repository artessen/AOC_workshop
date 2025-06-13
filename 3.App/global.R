# ==============================================================================
# Loading the required libraries ===============================================
library(caret)
library(xgboost)
library(ggplot2)
library(lattice)
library(stringr)
library(matrixStats)
library(tidyverse)

# ==============================================================================
# Loading the required libraries for plots======================================
library(RColorBrewer)
library(sf)
#library(rnaturalearth)
library(nestedcv)
library(reshape2)
library(stringr)
library(ggbeeswarm)

# ==============================================================================
# Loading the required libraries: GUIDO=========================================
library("readr")

# ==============================================================================
#for shiny
library("shiny")
library("bslib")
library("shinyFeedback")
library("shinyWidgets")
library("waiter")

options(browser="firefox")

# ==============================================================================
# Defining the custom objective and metric functions ===========================
source("00.a.aux_AuxiliaryFunctions.R")

# ==============================================================================
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

all_feature_names<-c(sERA5mean,sERA5anom,sERA5spec,sSPI,sCEMS)

# Areas of concern
sTgts = c("ColdS", "Drought", "HeatW", "HotDry", "RainD", "RainS", "TSumS", "TSumD")

sIDs = c("X","Y","Time","Month","Year","Day")

# Areas of concern: params for map
sfGrid = read_sf("AOC_grid_clean_4326.gpkg")
world = st_geometry(st_read("world","world"))
sVars = c("ColdS", "HeatW", "Drought", "HotDry", "RainD", "RainS", "TSumD", "TSumS")
sNames = c("Cold spell", "Heat wave", "Drought", "Hot and dry conditions", 
           "Rain deficit", "Rain excess", 
           "Temperature accumulation deficit", "Temperature accumulation surplus")
sCols = list(brewer.pal(9,"BuPu"), brewer.pal(9,"Reds"), brewer.pal(9,"Oranges"), brewer.pal(9,"OrRd"), 
             brewer.pal(9,"YlOrRd"), brewer.pal(9,"Blues"), brewer.pal(9,"GnBu"), brewer.pal(9,"YlOrBr"))
sBrk = c(0, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0)
limits = c(0,1)