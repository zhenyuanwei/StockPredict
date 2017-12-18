from app.dlmodels.dataengine import getHistoryData, convert_dataset
from app.dlmodels.stockmodel import train, predict, timestep, days

code = '000001'
start = '2000-01-01'
end = '2017-12-15'
if __name__ == '__main__':
    data = getHistoryData(start=start, end=end, code=code)
    data = convert_dataset(data, n_input=timestep, n_out=days)
    train(data=data, code=code)
    # result = predict(data, code=code)
