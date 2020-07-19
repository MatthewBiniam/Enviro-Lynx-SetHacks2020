#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd

from sklearn import linear_model
from predictor import predict

def get_data():
    land_data_path = '../Enviro Lynx Data/WaterData.csv'
    df = pd.read_csv(land_data_path)

    # Reuse of x-axis
    years = np.array(df['Year'])
    ryears = years.reshape(-1,1) # Regression model needs a 2D column vector

    return df, years, ryears

def ocean_temperatures():
    # Ocean Temperature
    temp = df['Temp (C)']
    temp = temp.replace(-999., 25)
    ocean_temp = np.array(temp)

    # Linear model
    reg = linear_model.LinearRegression()
    reg.fit(ryears, ocean_temp )

    years, value = predict(ocean_temp, reg)

    return years, value


def ph_scale():
    ph = df['PH scale']
    ph = ph.replace(-999., 8.1)
    ph_arr = np.array(ph)

    # Linear model
    reg = linear_model.LinearRegression()
    reg.fit(ryears, ph_arr)

    years, value = predict(ph_arr, reg)

    return years, value


def seawater_carbon():
    seawater = df.iloc[:, 4] # ppm
    seawater = seawater.replace(-999., 330)
    seawater_carbon = np.array(seawater)

    # Linear model
    reg = linear_model.LinearRegression()
    reg.fit(ryears, seawater_carbon)

    years, value = predict(seawater_carbon)

    return years, values

#print(seawater_carbon())
print(ocean_temperatures())
print(ph_scale())
print(seawater_carbon())
