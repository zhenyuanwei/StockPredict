from tushare import get_k_data
from pandas import DataFrame
from pandas import concat

def getHistoryData(start='2000-01-01',end='2017-12-15', code='000001'):
    df = get_k_data(code=code, start=start, end=end, index=True)
    df.set_index('date', inplace=True)
    df.drop(['code'], axis=1, inplace=True)
    return df

def convert_dataset(data, n_input=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = [], []
    # 输入序列 (t-n, ... t-1)
    for i in range(n_input, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # 输出结果 (t)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # 合并输入输出序列
    result = concat(cols, axis=1)
    result.columns = names
    # 删除包含缺失值的行
    if dropnan:
        result.dropna(inplace=True)
    return result

if __name__ == '__main__':
    historyDF = getHistoryData()
    historyDF = convert_dataset(historyDF, n_input=20)
    print(historyDF)