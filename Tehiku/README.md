# Tehiku dataset 

The Tehiku dataset includes both the time series and remote sensing datasets

## Time series 

The Tehiku-Time series dataset can be found from [value](https://github.com/ljj-cyber/RS4TS/tree/master/Data/Tehiku/value).

This dataset originates from soil moisture monitoring conducted by sensors embedded at various depths(10cm, 30cm, 60cm, 100cm) in the Tehiku forest region. This dataset spans the period from 2021-12-10 to 2022-10-1 in Tehiku, Kaitaia, New Zealand. The temporal granularity is 5 mins.

## Remote sensing

The Tehiku-Remote sensing dataset can be downloaded from [here](https://github.com/xren451/RSTS-interpolation). Download and put it into the Tehiku/RSvalue folder.

The original source is from [here](https://data.tpdc.ac.cn/en/data/c26201fc-526c-465d-bae7-5f02fa49d738/).

Formating: The soil moisture data is stored in netcdf format and Tiff format. File name: the file name is“ yyyyddd.nc ” or “ yyyyddd.tif ”, where yyyy stands for year and ddd stands for Julian date. For example, 2003001.nc represents this document describe the global soil moisture distribution on the first day of 2003. Data Projection: The data is EASE-grid2 equal-area projection data (with varying latitude and longitude intervals)， rather than usual equal-latitude-longitude data. (for more information about EASE-grid2 projection, please see https://nsidc.org/data/ease/ease_grid2.html.) The NC file of data stores three variables: latitude matrix, longitude matrix and soil moisture matrix, which are latitude (406*1), longitude（964*1） and soil_moisture (406*964) respectively. Projection information is not stored.
