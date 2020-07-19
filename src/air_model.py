#TODO
# - Fix minimum years based on the data set
# - Consider refactoring predict() as a decorator
#!/usr/bin/env python
# coding: utf-8
import pickle
import numpy as np
import pandas as pd

from sklearn import linear_model
from predictor import predict

def get_data():
    """
    ret: dataframe df, ndarray years, ndarray ryears
    ryears: reshaped 1D vector to 2D matrix, sklearn requires 2D
    years: used for x-axis
    """
    air_data_path = '../Enviro Lynx Data/AirData.csv'
    df = pd.read_csv(air_data_path)

    # Reuse of x-axis
    years = np.array(df['Year'])
    ryears = years.reshape(-1,1) # Regression model needs a 2D column vector

    return df, years, ryears


def population():
    """
    ret: int year, float value
    Value is humans per kilometre squared (based off land surface area)

    """
    df , years, ryears = get_data()
    population_data = np.array(df['Population Density'])

    # Training model
    reg = linear_model.LinearRegression()
    reg.fit(ryears, population_data)
    year, value = predict(population_data, reg)

    return year, value

def carbon_dioxide():
    """
    ret: int year, float value
    Value is CO2 emissions in kilotonnes in a year

    """
    df , years, ryears = get_data()
    carbon = np.array(df['CO2 emissions (kt)'])

    # Training and fitting
    reg = linear_model.LinearRegression()
    reg.fit(ryears, carbon)
    year, value = predict(carbon, reg)

    return year, value

def methane():
    """
    ret: int year, float value
    Value is methane emissions in kilotonnes in a year

    """
    df , years, ryears = get_data()
    methane = np.array(df['Methane emissions (kt)'])

    # Training and fitting
    reg = linear_model.LinearRegression()
    reg.fit(ryears, methane)
    year, value = predict(methane, reg)

    return year, value

def nitrogen_dioxide():
    df , years, ryears = get_data()
    # Methane emissions 
    nox = np.array(df['Nitrous oxide emissions(kt)'])

    # Training and fitting
    reg = linear_model.LinearRegression()
    reg.fit(ryears, nox)
    year, value = predict(nox, reg)

    return year, value

#if '__name__' == '__main__':
print(population())
print(carbon_dioxide())
print(methane())
print(nitrogen_dioxide())
