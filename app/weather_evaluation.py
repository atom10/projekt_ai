import requests
import datetime
import numpy as np

# współrzędne największych kopalni
mines = {
    'molybdenum': [
        (40.80833, 120.50000),  # Yangjiazhangzi
        (47.35023, 128.54517),  # Yichun Luming
        (31.62137, 116.91273),  # Shapinggou
        (34.33104, 109.95467),  # Jinduicheng
        (43.50000, 126.30000)   # Daheishan
    ],
    'manganese': [
        (27.1292, 22.8380),    # Nchwaning
        (27.2000, 22.9667),    # Wessels
        (27.1333, 22.8333),    # Black Rock
        (27.3766, 22.9813),    # Mamatwan
        (27.39230, 22.96862)   # Tshipi Borwa
    ]
}

def fetch_weather_data(lat, lon, date):
    url = f"https://api.open-meteo.com/v1/forecast"
    dt = datetime.datetime.strptime(date, "%d-%m-%Y")
    date_str = dt.strftime("%Y-%m-%dT%H:%M:%S")

    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,precipitation",
        "start": date_str,
        "end": date_str,
    }

    response = requests.get(url, params=params)

    if response.status_code == 200:
        data = response.json()
        if 'hourly' in data:
            hourly_data = data['hourly']
            # obliczanie średniej
            temperature = np.mean(hourly_data['temperature_2m'])
            precipitation = np.mean(hourly_data['precipitation']) if 'precipitation' in hourly_data else 0.0

            features = [temperature, precipitation]
            return features
    return None

def calculate_climate_averages(lat, lon):
    years = [f"{year}-06-01" for year in range(2015, 2023)]  # zakres dat dla średniej
    weather_data = []

    for date in years:
        features = fetch_weather_data(lat, lon, date)
        if features is not None:
            weather_data.append(features)

    if weather_data:
        weather_data = np.array(weather_data)
        average_values = np.mean(weather_data, axis=0)
        return average_values.tolist()
    else:
        return [22.0, 0.1]  # w przypadku niepowodzenia pobrania danych

def average_weather_region(lat, lon):
    return calculate_climate_averages(lat, lon)

def weather_evaluation(mineral, date):
    if mineral not in mines:
        raise ValueError("Unknown mineral")

    weather_features = []

    for coordinates in mines[mineral]:
        lat, lon = coordinates
        features = fetch_weather_data(lat, lon, date)

        if features is None:
            features = average_weather_region(lat, lon)

        weather_features.extend(features)

    return weather_features

#mineral = "molybdenum"
#date = "2024-07-07"
#weather_features = weather_evaluation(mineral, date)
#print(weather_features)