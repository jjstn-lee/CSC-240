import math
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from geopy.geocoders import Nominatim
from geopy.distance import geodesic as GD

def read_data():
    dataframe = pd.read_csv("adult.data")
    dataframe.columns = ["age", "work-class", "fnlwgt", "education", "education-num",
                        "martial-status", "occupation", "relationship",
                        "race", "sex", "capital-gain", "capital-loss",
                        "hours-per-week", "native-country", "wage"]
    mms = MinMaxScaler()
    # normalizes the numerical values in the dataframe
    dataframe["age"] = mms.fit_transform(dataframe[["age"]])
    dataframe["fnlwgt"] = mms.fit_transform(dataframe[["fnlwgt"]])
    dataframe["education-num"] = mms.fit_transform(dataframe[["education-num"]])
    dataframe["capital-gain"] = mms.fit_transform(dataframe[["capital-gain"]])
    dataframe["capital-loss"] = mms.fit_transform(dataframe[["capital-loss"]])
    dataframe["hours-per-week"] = mms.fit_transform(dataframe[["hours-per-week"]])
    return dataframe


def nominal(x, y):
    return int(x == y)
    

def numeric(x, y):
    total = ((x - y) ** 2)
    return 1 - math.sqrt(total)

def ordinal(x, y):
    a = x.split("-")
    b = y.split("-")
    intersection = len(list(set(a).intersection(set(b))))
    union = (len(a) + len(b)) - intersection
    return float(intersection) / union

def geo_dist(x, y):
    if x == y: return 1

    geolocator = Nominatim(user_agent="GoogleV3")

    spec_country = {
        "Outlying-US(Guam-USVI-etc)": "Guam",
        "Trinadad&Tobago": "Trinidad" }

    x_country = geolocator.geocode(x if x not in spec_country else spec_country[x])
    y_country = geolocator.geocode(y if y not in spec_country else spec_country[y])
    
    x_coord = (x_country.latitude, x_country.longitude)
    y_coord = (y_country.latitude, y_country.longitude)
    
    # The furthest you can get from another point on Earth is
    # half its circumference (km)
    furthest = 40075/2

    return 1 - ((GD(x_coord, y_coord).km)/furthest)


def similarity(x, y):
    sims = [
        numeric(x[0], y[0]),    # Age
        ordinal(x[1], y[1]),    # Workclass
        numeric(x[2], y[2]),    # Fnlwgt
        ordinal(x[3], y[3]),    # Education
        numeric(x[4], y[4]),    # Education-num
        ordinal(x[5], y[5]),    # Marital-status
        nominal(x[6], y[6]),    # Occupation
        nominal(x[7], y[7]),    # Relationship
        nominal(x[8], y[8]),    # Race
        nominal(x[9], y[9]),    # Sex
        numeric(x[10], y[10]),  # Capital-gain
        numeric(x[11], y[11]),  # Capital-loss
        numeric(x[12], y[12]),  # Hours-per-week
        geo_dist(x[13], y[13])  # Native-country
    ]
    return sum(sims) / len(sims)


def main():
    df = read_data()
    row1 = df.loc[0]
    row2  = df.loc[1]
    print(f"Final distance: {similarity(row1, row2)}")

main()