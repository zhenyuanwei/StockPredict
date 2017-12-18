from app.dlmodels.stockmodel import predict, timestep, feature, days
from app.dlmodels.dataengine import getHistoryData, convert_dataset
import time
from sklearn.preprocessing import MinMaxScaler

def getToday():
    today = time.localtime()
    value = time.strftime('%Y-%m-%d', today)
    return value

def getData(end, code='000001'):
    data = getHistoryData(end=end, code=code)
    data = convert_dataset(data, n_input=timestep, n_out=1)
    data = data.values
    data = data[-1:, feature:]
    return data

def inverse(values, code='000001'):
    data = getHistoryData(code=code)
    data = convert_dataset(data, n_input=timestep, n_out=days)
    scaler = MinMaxScaler().fit(data)
    results = scaler.inverse_transform(values)
    return results

def predictValue(code='000001'):
    today = getToday()
    data = getData(end=today, code=code)
    results = predict(data, code=code)
    results = inverse(results, code=code)
    return results

if __name__ == '__main__':
    results = predictValue()
