#!/usr/bin/env python
# coding: utf-8

def predict(data, model, vals):
    """
    ret: int year, float value
    Finds the year when a given parameter will be reacher
    Finds the value of an environmental issue at a given year
    """
    minimum = int(data[-1] + 1)
    find_year_inp = vals[0]
    find_val_inp = vals[1]

    # Only allows for future predictions
    try:
        assert find_year_inp > minimum
    except:
        print(f'Please enter a value larger than {minimum}')
        value = 0
        return None

    try:
        assert find_val_inp >= 2020
    except:
        print(f'Please enter a year larger than 2019')
        year = 0
        return None

    year = int(((find_year_inp - model.intercept_) / model.coef_)[0]) # Given user specified input value
    value = model.predict([[find_val_inp]])[0] # Given user specified year

    return year, value
