#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import linear_model

def get_data():
    air_data_path = '../Enviro Lynx Data/AirData.csv'
    df = pd.read_csv(air_data_path)

    # Reuse of x-axis
    years = np.array(df['Year'])
    ryears = years.reshape(-1,1) # Regression model needs a 2D column vector
    hundred_years = np.arange(2020,2121).reshape(-1,1)

    return df, years, ryears, hundred_years

def predict(data, model):
    """
    ret: int year, float value
    Finds the year when a given parameter will be reacher
    Finds the value of an environmental issue at a given year
    """
    minimum = int(data[-1] + 1)
    find_year_inp = float(input(f'Enter a value greater than {minimum}\n'))
    find_val_inp = int(input(f'Enter a year greater than 2019\n'))

    # Only allows for future predictions
    try:
        assert find_year_inp > minimum
    except:
        print(f'Please enter a value larger than {minimum}')
        find_year_inp = float(input(f'Enter a value greater than {minimum}\n'))

    try:
        assert find_val_inp >= 2020
    except:
        print(f'Please enter a year larger than 2019')
        find_val_inp = int(input(f'Enter a year greater than 2019\n'))

    year = int(((find_year_inp - model.intercept_) / model.coef_)[0]) # Given user specified input value
    value = model.predict([[find_val_inp]])[0] # Given user specified year

    return year, value

def population():
    """
    ret: int year, float value
    Value is humans per kilometre squared (based off land surface area)

    """
    df , years, ryears, hundred_years = get_data()
    population = np.array(df['Population Density'])

    # Training model
    reg = linear_model.LinearRegression()
    reg.fit(ryears, population)
    year, value = predict(population, reg)

    return year, value

def carbon_dioxide():
    """
    ret: int year, float value
    Value is CO2 emissions in kilotonnes in a year

    """
    df , years, ryears, hundred_years = get_data()
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
    df , years, ryears, hundred_years = get_data()
    methane = np.array(df['Methane emissions (kt)'])

    # Training and fitting
    reg = linear_model.LinearRegression()
    reg.fit(ryears, methane)
    year, value = predict(methane, reg)

    return year, value

def nitrogen_dioxide():
    df , years, ryears, hundred_years = get_data()
    # Methane emissions 
    nox = np.array(df['Nitrous oxide emissions(kt)'])

    # Training and fitting
    reg = linear_model.LinearRegression()
    reg.fit(ryears, nox)
    year, value = predict(nox, reg)

    return year, value

if '__name__' == '__main__':
    print(population())
    print(carbon_dioxide())
    print(methane())
    print(nitrogen_dioxide())
