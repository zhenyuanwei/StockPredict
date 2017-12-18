from torch import nn
from torch import save as save_model
from torch import load as load_model
from torch.autograd import Variable
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import torch
from pandas import DataFrame

feature = 5
timestep = 20
days = 5
batch_size = 64
epochs = 200

def createTensorDataset(data):
    values = data.values
    x = values[:, : feature * timestep - 1]
    x = torch.FloatTensor(x)

    y = values[:, feature * timestep :]
    y = np.reshape(y, (y.shape[0], 1)).astype('float32')
    y = torch.from_numpy(y)

    dataset = TensorDataset(data_tensor=x, target_tensor=y)
    return dataset


class StockModel(nn.Module):

    def __init__(self):
        super(StockModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size=feature * timestep, hidden_size=400, num_layers=5, dropout=0.2)
        self.linear1 = nn.Linear(in_features=400, out_features=1200)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(in_features=1200, out_features=feature * days)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = out[:, -1]
        out = self.linear1(out)
        out = self.relu(out)
        out = self.linear2(out)
        return out

def train(data, code='000001', folder='../dlmodels'):
    model_file = folder + '/' + 'model_' + code + '.ptm'
    model = StockModel()
    dataset = createTensorDataset(data)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(params=model.parameters())

    dataloader = DataLoader(dataset=dataset, batch_size=batch_size)
    model.train()
    for epoch in range(epochs):
        lossess = []
        for (data, label) in dataloader:
            data = Variable(data)
            label = Variable(label)

            output = model(data)

            loss = criterion(output, label)
            loss_data = loss.data[0]
            lossess.append(loss_data)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('epoch {}/{} loss: {:.6f}'.format(epoch + 1, epochs, np.mean(lossess)))

    save_model(model, model_file)

def format_outputs(outputs):
    outputs = np.reshape(outputs, (feature, days))
    df = DataFrame(outputs)
    df.columns = ['open', 'close', 'high', 'low', 'volume']
    return df

def predict(data, code='000001', folder='../dlmodels'):
    model_file = folder + '/' + 'model_' + code + '.ptm'

    model = load_model(model_file)
    model.eval()
    dataset = torch.FloatTensor(data)
    x = Variable(dataset)
    outputs = model(x)
    outputs = outputs.data.numpy()
    return format_outputs(outputs)