#@title Wind gust
#from lightgbm.sklearn import LGBMClassifier
from lightgbm.sklearn import LGBMRegressor
import streamlit as st
import numpy as np
import pandas as pd
import pickle
import plotly.express as px
import matplotlib.pyplot as plt
import requests
import json
import plotly.express as px
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
from datetime import timedelta
from sklearn.metrics import accuracy_score
import sklearn

def get_wind(st_id):
  #get actual des dir
  des_dir = requests.get("https://servizos.meteogalicia.gal/mgrss/observacion/ultimos10minEstacionsMeteo.action?idEst="+st_id+"&idParam=DV_SD_10m")
  json_data = json.loads(des_dir.content)
  instant = json_data['listUltimos10min'][0]['instanteLecturaUTC']
  des_dir = json_data['listUltimos10min'][0]['listaMedidas'][0]['valor']

  #get actual dir
  dir = requests.get("https://servizos.meteogalicia.gal/mgrss/observacion/ultimos10minEstacionsMeteo.action?idEst="+st_id+"&idParam=DV_AVG_10m")
  json_data = json.loads(dir.content)
  dir = json_data['listUltimos10min'][0]['listaMedidas'][0]['valor']

  #get actual mod
  mod = requests.get("https://servizos.meteogalicia.gal/mgrss/observacion/ultimos10minEstacionsMeteo.action?idEst="+st_id+"&idParam=VV_AVG_10m")
  json_data = json.loads(mod.content)
  mod = json_data['listUltimos10min'][0]['listaMedidas'][0]['valor']*1.94384

  #get actual des mod
  des_mod = requests.get("https://servizos.meteogalicia.gal/mgrss/observacion/ultimos10minEstacionsMeteo.action?idEst="+st_id+"&idParam=VV_SD_10m")
  json_data = json.loads(des_mod.content)
  des_mod = json_data['listUltimos10min'][0]['listaMedidas'][0]['valor']*1.94384

  return instant, dir, des_dir,mod, des_mod





def get_meteogalicia_model_4Km(coorde):
    """
    get meteogalicia model (4Km) from algo coordenates
    Returns
    -------
    dataframe with meteeorological variables forecasted.
    """

    #defining url to get model from Meteogalicia server
    today=pd.to_datetime("today")

    try:

      head1="http://mandeo.meteogalicia.es/thredds/ncss/modelos/WRF_HIST/d03"
      head2=today.strftime("/%Y/%m/wrf_arw_det_history_d03")
      head3=today.strftime("_%Y%m%d_0000.nc4?")
      head=head1+head2+head3

      var1="var=dir&var=mod&var=wind_gust&var=mslp&var=temp&var=rh&var=visibility&var=lhflx"
      var2="&var=lwflx&var=conv_prec&var=prec&var=swflx&var=shflx&var=cape&var=cin&var=cfh&var=T850"
      var3="&var=cfl&var=cfm&var=cft&var=HGT500&var=HGT850&var=T500&var=snow_prec&var=snowlevel"
      var=var1+var2+var3

      f_day=(today+timedelta(days=3)).strftime("%Y-%m-%d")
      tail="&time_start="+today.strftime("%Y-%m-%d")+"T01%3A00%3A00Z&time_end="+f_day+"T23%3A00%3A00Z&accept=csv"

      dffinal=pd.DataFrame()
      for coor in list(zip(coorde.lat.tolist(),coorde.lon.tolist(),np.arange(0,len(coorde.lat.tolist())).astype(str))):
          dffinal=pd.concat([dffinal,pd.read_csv(head+var+"&latitude="+str(coor[0])+"&longitude="+str(coor[1])+tail,).add_suffix(str(coor[2]))],axis=1)


      #filter all columns with lat lon and date
      dffinal=dffinal.filter(regex='^(?!(lat|lon|date).*?)')

      #remove column string between brakets
      new_col=[c.split("[")[0]+c.split("]")[-1] for c in dffinal.columns]
      for col in zip(dffinal.columns,new_col):
          dffinal=dffinal.rename(columns = {col[0]:col[1]})

      dffinal=dffinal.set_index(pd.date_range(start=today.strftime("%Y-%m-%d"), end=(today+timedelta(days=4)).strftime("%Y-%m-%d"), freq="H")[1:-1])

    except:

      today  = pd.to_datetime("today")-timedelta(1)
      head1="http://mandeo.meteogalicia.es/thredds/ncss/modelos/WRF_HIST/d03"
      head2=today.strftime("/%Y/%m/wrf_arw_det_history_d03")
      head3=today.strftime("_%Y%m%d_0000.nc4?")
      head=head1+head2+head3

      var1="var=dir&var=mod&var=wind_gust&var=mslp&var=temp&var=rh&var=visibility&var=lhflx"
      var2="&var=lwflx&var=conv_prec&var=prec&var=swflx&var=shflx&var=cape&var=cin&var=cfh&var=T850"
      var3="&var=cfl&var=cfm&var=cft&var=HGT500&var=HGT850&var=T500&var=snow_prec&var=snowlevel"
      var=var1+var2+var3

      f_day=(today+timedelta(days=3)).strftime("%Y-%m-%d")
      tail="&time_start="+today.strftime("%Y-%m-%d")+"T01%3A00%3A00Z&time_end="+f_day+"T23%3A00%3A00Z&accept=csv"

      dffinal=pd.DataFrame()
      for coor in list(zip(coorde.lat.tolist(),coorde.lon.tolist(),np.arange(0,len(coorde.lat.tolist())).astype(str))):
          dffinal=pd.concat([dffinal,pd.read_csv(head+var+"&latitude="+str(coor[0])+"&longitude="+str(coor[1])+tail,).add_suffix(str(coor[2]))],axis=1)


      #filter all columns with lat lon and date
      dffinal=dffinal.filter(regex='^(?!(lat|lon|date).*?)')

      #remove column string between brakets
      new_col=[c.split("[")[0]+c.split("]")[-1] for c in dffinal.columns]
      for col in zip(dffinal.columns,new_col):
          dffinal=dffinal.rename(columns = {col[0]:col[1]})

      dffinal=dffinal.set_index(pd.date_range(start=today.strftime("%Y-%m-%d"), end=(today+timedelta(days=4)).strftime("%Y-%m-%d"), freq="H")[1:-1])


    return dffinal



def get_meteogalicia_model_1Km(coorde):
    """
    get meteogalicia model (1Km)from algo coordenates
    Returns
    -------
    dataframe with meteeorological variables forecasted.
    """

    #defining url to get model from Meteogalicia server
    var1 = "var=dir&var=mod&var=wind_gust&var=mslp&var=temp&var=rh&var=visibility&var=lhflx"
    var2 = "&var=lwflx&var=conv_prec&var=prec&var=swflx&var=shflx&var=cape&var=cin&var=cfh&var=T850"
    var3 = "&var=cfl&var=cfm&var=cft&var=HGT500&var=HGT850&var=T500&var=snow_prec&var=snowlevel"
    var = var1+var2+var3
    head1 = "https://mandeo.meteogalicia.es/thredds/ncss/wrf_1km_baixas/fmrc/files/"

    try:

      today = pd.to_datetime("today")
      head2 = today.strftime("/%Y%m%d/wrf_arw_det1km_history_d05")
      head3 = today.strftime("_%Y%m%d_0000.nc4?")
      head = head1+head2+head3

      f_day=(today+timedelta(days=2)).strftime("%Y-%m-%d")
      tail="&time_start="+today.strftime("%Y-%m-%d")+"T01%3A00%3A00Z&time_end="+f_day+"T23%3A00%3A00Z&accept=csv"

      dffinal=pd.DataFrame()
      for coor in list(zip(coorde.lat.tolist(),coorde.lon.tolist(),np.arange(0,len(coorde.lat.tolist())).astype(str))):
          dffinal=pd.concat([dffinal,pd.read_csv(head+var+"&latitude="+str(coor[0])+"&longitude="+str(coor[1])+tail,).add_suffix(str(coor[2]))],axis=1)

      #filter all columns with lat lon and date
      dffinal=dffinal.filter(regex='^(?!(lat|lon|date).*?)')

      #remove column string between brakets
      new_col=[c.split("[")[0]+c.split("]")[-1] for c in dffinal.columns]
      for col in zip(dffinal.columns,new_col):
          dffinal=dffinal.rename(columns = {col[0]:col[1]})

      dffinal=dffinal.set_index(pd.date_range(start=today.strftime("%Y-%m-%d"), end=(today+timedelta(days=3)).strftime("%Y-%m-%d"), freq="H")[1:-1])
      control = True

    except:

      today = pd.to_datetime("today")-timedelta(1)
      head2 = today.strftime("/%Y%m%d/wrf_arw_det1km_history_d05")
      head3 = today.strftime("_%Y%m%d_0000.nc4?")
      head = head1+head2+head3

      f_day=(today+timedelta(days=2)).strftime("%Y-%m-%d")
      tail="&time_start="+today.strftime("%Y-%m-%d")+"T01%3A00%3A00Z&time_end="+f_day+"T23%3A00%3A00Z&accept=csv"

      dffinal=pd.DataFrame()
      for coor in list(zip(coorde.lat.tolist(),coorde.lon.tolist(),np.arange(0,len(coorde.lat.tolist())).astype(str))):
          dffinal=pd.concat([dffinal,pd.read_csv(head+var+"&latitude="+str(coor[0])+"&longitude="+str(coor[1])+tail,).add_suffix(str(coor[2]))],axis=1)


      #filter all columns with lat lon and date
      dffinal=dffinal.filter(regex='^(?!(lat|lon|date).*?)')

      #remove column string between brakets
      new_col=[c.split("[")[0]+c.split("]")[-1] for c in dffinal.columns]
      for col in zip(dffinal.columns,new_col):
          dffinal=dffinal.rename(columns = {col[0]:col[1]})

      dffinal=dffinal.set_index(pd.date_range(start=today.strftime("%Y-%m-%d"), end=(today+timedelta(days=3)).strftime("%Y-%m-%d"), freq="H")[1:-1])
      control= False


    return dffinal

options = ["marin", "udra", "ons","coron"]
default_option = options[0]  # Set the default option

# Create a radio button to select the string variable
station = st.radio("Select airport", options, index=0)

#stations_id
station_id = {"marin":"14005","ons":"10126","udra":"10905","coron":"10085"}

#load algorithm file gust

#load algorithm file gust
algo_g_d0 = pickle.load(open(station+"algorithms/gust_"+station+"_d0.al","rb"))
algo_g_d1 = pickle.load(open(station+"algorithms/gust_"+station+"_d1.al","rb"))
algo_g_d2 = pickle.load(open(station+"algorithms/gust_"+station+"_d2.al","rb"))


meteo_model = get_meteogalicia_model_1Km(algo_g_d0["coor"])

#add time variables
meteo_model["hour"] = meteo_model.index.hour
meteo_model["month"] = meteo_model.index.month
meteo_model["dayofyear"] = meteo_model.index.dayofyear
meteo_model["weekofyear"] = meteo_model.index.isocalendar().week.astype(int)

#get station dat
r_gust = requests.get("https://servizos.meteogalicia.gal/mgrss/observacion/ultimosHorariosEstacions.action?idEst="+station_id[station]+"&idParam=VV_RACHA_10m&numHoras=36")
json_data = json.loads(r_gust.content)

gust_o, time = [],[]
for c in json_data["listHorarios"]:
  for c1 in c['listaInstantes']:
    time.append(c1['instanteLecturaUTC'])
    gust_o.append(c1['listaMedidas'][0]["valor"])

df_st = pd.DataFrame(np.array(gust_o),columns=["observed_gust"],index= time)
df_st.index = pd.to_datetime(df_st.index )

#select x _var
model_x_var_p0 = meteo_model[:24][algo_g_d0["x_var"]]
model_x_var_p1 = meteo_model[24:48][algo_g_d1["x_var"]]
model_x_var_p2 = meteo_model[48:72][algo_g_d2["x_var"]]


#forecast machine learning gust
gust_ml0 = algo_g_d0["pipe"].predict(model_x_var_p0)
gust_ml1 = algo_g_d1["pipe"].predict(model_x_var_p1)
gust_ml2 = algo_g_d2["pipe"].predict(model_x_var_p2)



#compare results
df_mod=pd.DataFrame({"time":meteo_model[:96].index,
                      "ML_gust": np.concatenate((gust_ml0,gust_ml1,gust_ml2),axis=0),
                      "WRF_gust": meteo_model.wind_gust1})


df_res = pd.concat([df_mod.set_index("time"),df_st],axis=1).dropna()
mae_ml = round(mean_absolute_error(df_res["observed_gust"],df_res["ML_gust"]),2)
mae_wrf = round(mean_absolute_error(df_res["observed_gust"],df_res["WRF_gust"]),2)

if mae_ml < mae_wrf:
  score_ml+=1
if mae_ml > mae_wrf:
  score_wrf+=1

fig, ax = plt.subplots(figsize=(10,6))
df_res = round(df_res*1.94384,0)
df_res.plot(grid=True, ax=ax, color=["b","r","g"], linestyle='--');
ref_met = algo_g_d0["score"]["MAE_met"]
ref_ml = algo_g_d0["score"]["MAE_ml"]
ax.set_title("{} wind gust max hour before (knots)\nActual MAE (m/s)  meteorological model (point 1): {}. Reference: {}\nActual MAE (m/s) machine learning: {}. Reference: {}".format(station_name,mae_wrf,ref_met,mae_ml,ref_ml))
plt.grid(True, which = "both", axis = "both")
#plt.show()
st.pyplot(fig)

df_mod = df_mod.set_index("time")
df_mod = round(df_mod*1.94384,0)
fig, ax = plt.subplots(figsize=(10,6))
df_mod[:24].plot(grid=True,ax=ax,color=["b","r"]);
ax.set_title("{} wind gust max hour before day=0 (knots)\nMAE (m/s) meteorological model (point 1): {}\nMAE (m/s) machine learning: {}".format(station_name,ref_met,ref_ml))
plt.grid(True, which = "both", axis = "both")
#plt.show()
st.pyplot(fig)

fig, ax = plt.subplots(figsize=(10,6))
df_mod[24:48].plot(grid=True,ax=ax,color=["b","r"]);
ref_met = algo_g_d1["score"]["MAE_met"]
ref_ml = algo_g_d1["score"]["MAE_ml"]
ax.set_title("{} wind gust max hour before day=1 (knots)\nMAE (m/s) meteorological model (point 1): {}\nMAE (m/s) machine learning: {}".format(station_name,ref_met,ref_ml))
plt.grid(True, which = "both", axis = "both")
#plt.show()
st.pyplot(fig)

fig, ax = plt.subplots(figsize=(10,6))
df_mod[48:72].plot(grid=True,ax=ax,color=["b","r"]);
ref_met = algo_g_d2["score"]["MAE_met"]
ref_ml = algo_g_d2["score"]["MAE_ml"]
ax.set_title("{} wind gust max hour before day=2 (knots)\nMAE (m/s) meteorological model (point 1): {}\nMAE (m/s) machine learning: {}".format(station_name,ref_met,ref_ml))
plt.grid(True, which = "both", axis = "both")
#plt.show()
st.pyplot(fig)
