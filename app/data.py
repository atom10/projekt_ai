from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import political_evaluation as pe
import weather_evaluation as we
import stock_evaluation as se
import logistic_evaluation as le

minerals = {
    'manganese': 0.0,
    'molybdenum': 1.0
}

def generate_date_range(date_start_str, date_end_str):
    date_start = datetime.strptime(date_start_str, "%d-%m-%Y")
    date_end = datetime.strptime(date_end_str, "%d-%m-%Y")
    delta = (date_end - date_start).days
    return [date_start + timedelta(days=i) for i in range(delta + 1)]

def generate_date_range_from_today(n):
    start_date = datetime.now()
    date_list = [start_date + timedelta(days=x) for x in range(n+1)]
    return date_list

def generate_singe_data_packet(current_date, mineral, step=3, with_target=True, generated_data=None):
    if generated_data is not None:
        timestamp = float(current_date.timestamp())
        existing_data = generated_data[
            (generated_data[:, 1] == timestamp) & (generated_data[:, 0] == minerals[mineral])]
        if existing_data.size > 0:
            return existing_data[0]

    value = minerals[mineral]  # raw value of mineral
    if with_target:
        return np.concatenate((
            [value],
            [float(current_date.timestamp())],
            pe.evaluate(
                mineral,
                (current_date - timedelta(days=step)).strftime("%d-%m-%Y"),
                (current_date + timedelta(days=step)).strftime("%d-%m-%Y")
            ),
            we.weather_evaluation(mineral,current_date.strftime("%d-%m-%Y")),
            le.logistic_evaluation(current_date.strftime("%d-%m-%Y")),
            se.getValue(current_date.strftime("%d-%m-%Y"), mineral)
        ), axis=None)
    else:
        return np.concatenate((
            [value],
            [float(current_date.timestamp())],
            pe.evaluate(
                mineral,
                (current_date - timedelta(days=step)).strftime("%d-%m-%Y"),
                (current_date + timedelta(days=step)).strftime("%d-%m-%Y")
            ),
            we.weather_evaluation(mineral,current_date.strftime("%d-%m-%Y")),
            le.logistic_evaluation(current_date.strftime("%d-%m-%Y"))
        ), axis=None)

# date dd-mm-yyyy
def generate_data(date_from, date_to, step=3, existing_data=None):
    output = []
    # Convert date strings to datetime objects
    start_date = datetime.strptime(date_from, "%d-%m-%Y")
    end_date = datetime.strptime(date_to, "%d-%m-%Y")

    # Initialize the current date to the start date
    current_date = start_date

    # Loop from start_date to end_date with a step of 'step' days
    while current_date <= end_date:
        data_packet = []
        current_date += timedelta(days=step)
        for mineral, value in minerals.items():
            data_packet = generate_singe_data_packet(current_date, mineral, step, generated_data=existing_data)
            output.append(data_packet)
        print("generated data for " + current_date.strftime("%d-%m-%Y") + "  (" + str(
            "{:.2f}".format((current_date - start_date) / (end_date - start_date) * 100.0)) + "%)")
    return np.asarray(output)

delimiter = ";"

def save_data(data, output_name="data.csv"):
    np.savetxt(output_name, data, delimiter=delimiter)

def load_data(input_name="data.csv"):
    return np.genfromtxt(input_name, delimiter=delimiter)
