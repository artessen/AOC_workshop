# Wrapper function
# Loading the pre-defined function
source("00.a.fTrainModel.R")
source("00.b.fPredictModel.R")
source("00.c.fPlotResults.R")

# Select your Target Variable of interest:
#  Possible options are:
#    - ColdS    --> Cold Spell
#    - HeatW    --> Heatwave
#    - Drought  --> Agricultural Drought
#    - HotDry   --> Hot and dry conditions
#    - RainS    --> Rain accumulation surplus
#    - RainD    --> Rain accumulation deficit
#    - TSumS    --> Temperature accumulation surplus
#    - TSumD    --> Temperature accumulation deficit
Target = "HeatW"

# Select your Input Features of interest: 
# Select the list of weather-related variables (ERA5):
#  Possible options are:
#    - msl      --> Mean sea level pressure [Pa]
#    - sp       --> Surface pressure [Pa]
#    - t2m      --> Mean 2m temperature [K]
#    - tmax     --> Maximum 2m temperature [K]
#    - tmin     --> Minimum 2m temperature [K]
#    - tp       --> Total precipitation [m]
#    - z500     --> Geopotential at 500 hPa [m2 s-2]
Features_ERA5 = c("tmax","tmin")
# Select the list of AOC specific variables:
#  Possible options are:
#    - daystmax25a30C   --> Days with maximum daily temperature between 25 and 30 °C
#    - daystmax30C      --> Days with maximum daily temperature > 30 °C
#    - daystmin0C       --> Days with minimum daily temperature < 0 °C
#    - daystminm10C     --> Days with minimum daily temperature < -10 °C
#    - daystminm18C     --> Days with minimum daily temperature < -18 °C
#    - daystp01mm       --> Days with daily precipitation > 1 mm
#    - daystp30mm       --> Days with daily precipitation > 30 mm
Features_AOC = c("daystmax30C")
# Select the list of CEMS specific variables:
#  Possible options are:
#    - spi01     --> Standardized Precipitation Index, 1-month accumulation
#    - spi03     --> Standardized Precipitation Index, 3-months accumulation
#    - spi06     --> Standardized Precipitation Index, 6-months accumulation
#    - spi09     --> Standardized Precipitation Index, 9-months accumulation
#    - spi12     --> Standardized Precipitation Index, 12-months accumulation
#    - smant     --> Soil moisture index anomaly
#    - fpanv     --> fAPAR anomaly
Features_CEMS = NA #c("spi01", "spi06", "smant", "fpanv")
# Creating the list of your selection of Input Features of interest: 
Features = list(Features_ERA5, Features_AOC, Features_CEMS)

# Delete folder of AOC, if it exists already
for (sTgt in Target) {
  sDir = paste("./",sTgt,sep="")
  if (dir.exists(sDir)) { unlink(sDir, recursive=T) }
}

# Training the model for the selected features
fTrainModel(Target, Features)

# Using our trained model to predict the AOC conditions in August 2022
fPredictModel(Target)

# Evaluating the results of our model
fPlotResults(Target)

