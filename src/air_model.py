#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import isotonic

def get_data():
    air_data_path = '../data/AirData.csv'
    df = pd.read_csv(air_data_path)

    # Reuse of x-axis
    years = np.array(df['Year'])
    ryears = years.reshape(-1,1) # Regression model needs a 2D column vector
    hundred_years = np.arange(2020,2121).reshape(-1,1)

    return df, years, ryears, hundred_years


def population():
    df , years, ryears, hundred_years = get_data()
    population = np.array(df['Population Density'])

    # Training model
    reg = linear_model.LinearRegression()
    reg.fit(ryears, population)

    # Predict based off input
    minimum = int(population[0] + 1)
    inp = float(input(f'Enter a value greater than {minimum}\n'))

    try:
        assert inp > minimum
    except:
        print(f'There is no historical data for this input. Please enter a value larger than {minimum}')
        inp = float(input(f'Enter a value greater than {minimum}\n'))

    year = (inp - reg.intercept_) / reg.coef_

    if year[0] < 2020:
        return f'In the year {int(year[0])} there were {inp} humans per km^2'
    else:
        return f'By year {int(year[0])} there will be {inp} humans per km^2'

def carbon_dioxide():
    df , years, ryears, hundred_years = get_data()
    # Carbon emissions, kilotons
    carbon = np.array(df['CO2 emissions (kt)'])

    # Training and fitting
    reg = linear_model.LinearRegression()
    reg.fit(ryears, carbon)

    # Predict based off input
    minimum = int(carbon[0] + 1)
    inp = float(input(f'Enter a value greater than {minimum}\n'))

    try:
        assert inp > minimum
    except:
        print(f'There is no historical data for this input. Please enter a value larger than {minimum}')
        inp = float(input(f'Enter a value greater than {minimum}\n'))

    year = (inp - reg.intercept_) / reg.coef_

    if year[0] < 2020:
        return f'In the year {int(year[0])} there were {inp} kilotonnes of carbon emitted'
    else:
        return f'By year {int(year[0])} there will be {inp} kilotonnes of carbon emitted'


def methane():
    df , years, ryears, hundred_years = get_data()

    # Methane emissions 
    methane = np.array(df['Methane emissions (kt)'])
    fig = plt.figure(figsize=(1,1))
    ax = fig.add_axes([1,1,5,5])
    plot = ax.plot(years,methane)

    # Training and fitting
    reg = linear_model.LinearRegression()
    reg.fit(ryears, methane)

    # Predict based off input
    minimum = int(methane[0] + 1)
    inp = float(input(f'Enter a value greater than {minimum}\n'))

    try:
        assert inp > minimum
    except:
        print(f'There is no historical data for this input. Please enter a value larger than {minimum}')
        inp = float(input(f'Enter a value greater than {minimum}\n'))

    year = (inp - reg.intercept_) / reg.coef_

    if year[0] < 2020:
        return f'In the year {int(year[0])} there were {inp} kilotonnes of methane emitted'
    else:
        return f'By year {int(year[0])} there will be {inp} kilotonnes of methane emitted'

def nitrogen_dioxide():
    df , years, ryears, hundred_years = get_data()
    # Methane emissions 
    nox = np.array(df['Nitrous oxide emissions(kt)'])
    fig = plt.figure(figsize=(1,1))
    ax = fig.add_axes([1,1,5,5])
    plot = ax.plot(years,nox)

    # Training and fitting
    reg = linear_model.LinearRegression()
    reg.fit(ryears, nox)

    # Predict based off input
    minimum = int(nox[0] + 1)
    inp = float(input(f'Enter a value greater than {minimum}\n'))

    try:
        assert inp > minimum
    except:
        print(f'There is no historical data for this input. Please enter a value larger than {minimum}')
        inp = float(input(f'Enter a value greater than {minimum}\n'))

    year = (inp - reg.intercept_) / reg.coef_

    if year[0] < 2020:
        return f'In the year {int(year[0])} there were {inp} kilotonnes of nitrous oxide emitted'
    else:
        return f'By year {int(year[0])} there will be {inp} kilotonnes of nitrous oxide emitted'

    # Isotonic regression
    #iso = isotonic.IsotonicRegression() 
    #iso.fit(years, nox, sample_weight=2) # 1D array
    #plot = ax.plot(years, reg.predict(ryears))

print(population())
print(carbon_dioxide())
print(methane())
print(nitrogen_dioxide())
