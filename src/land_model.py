#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd

from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from predictor import predict


def get_data():
    land_data_path = '../Enviro Lynx Data/GroundData.csv'
    df = pd.read_csv(land_data_path)

    # Reuse of x-axis
    years = np.array(df['Year'])
    ryears = years.reshape(-1,1) # Regression model needs a 2D column vector

    return df, years, ryears


def forest_area():
    df, years, ryears = get_data()
    forest_area = np.array(df['Forest Area(sq . Km)'])

    # Model
    reg = linear_model.LinearRegression()
    reg.fit(ryears, forest_area)
    year, value = predict(forest_area, reg)

    return year, value


def urban_population():
    df, years, ryears = get_data()
    urban_population = np.array(df['Urban Population(people)'])

    # Polynomial model
    poly_model = PolynomialFeatures(3)
    model = make_pipeline(poly_model, linear_model.Ridge())
    model.fit(ryears, urban_population)
    # check to see if works


    return year, value

def agriculture_area():
    # World Agriculture Area
    agri_df = df['World Agriculture Area(%)']
    agriculture = np.array(agri_df)

    # Linear model, restricting to most recent results
    # Very large drop starting in 2018
    restricted_agri = agriculture[-10:]
    reg = linear_model.LinearRegression()
    reg.fit(ryears[-10:], restricted_agri)

    year, value = predict(agriculture, reg)

    return year, value

# Find a way to reuse
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

#if '__name__' == '__main__':
print(forest_area())
print(urban_population())
print(agriculture_area())
print(population())

