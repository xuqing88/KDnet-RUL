import torch
from torch import nn
from torch.utils.data import Dataset
import torch.nn.functional as F

class LSTM_RUL(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, dropout, bid, device):
        super(LSTM_RUL, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.bid = bid
        self.dropout = dropout
        self.device = device
        # encoder definition
        self.encoder = nn.LSTM(input_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True, bidirectional=self.bid)
        # regressor
        self.regressor= nn.Sequential(
            nn.Linear(self.hidden_dim+self.hidden_dim*self.bid, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.hidden_dim, 1))

    def forward(self, src):
        # input shape [batch_size, seq_length, input_dim]
        # outputs = [batch size, src sent len,  hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        encoder_outputs, (hidden, cell) = self.encoder(src)
#         encoder_outputs = F.dropout(torch.relu(encoder_outputs), p=0.5, training=self.training)
        # select the last hidden state as a feature
        features = encoder_outputs[:, -1:].squeeze()
        predictions = self.regressor(features)
        return predictions.squeeze(), features


class LSTM_RUL_Student_simple(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, dropout, bid, device):
        super(LSTM_RUL_Student_simple, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.bid = bid
        self.dropout = dropout
        self.device = device
        # encoder definition
        self.encoder = nn.LSTM(input_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True, bidirectional=self.bid)
        # adapter
        self.adapter = nn.Linear(self.hidden_dim*2,64) # output = teacher network feature output
        # self.adapter = nn.Conv1d(self.hidden_dim*2,self.hidden_dim*2,1)
        # regressor
        self.regressor= nn.Sequential(
            nn.Linear(self.hidden_dim*2, self.hidden_dim) ,
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.hidden_dim, 1))

    def forward(self, src):
        # input shape [batch_size, seq_length, input_dim]
        # outputs = [batch size, src sent len,  hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        encoder_outputs, (hidden, cell) = self.encoder(src)
#         encoder_outputs = F.dropout(torch.relu(encoder_outputs), p=0.5, training=self.training)
        # select the last hidden state as a feature
        features = encoder_outputs[:, -1:].squeeze()
        hint = self.adapter(features)
        predictions = self.regressor(features)
        return predictions.squeeze(), features, hint


class CNN_RUL_student(nn.Module):
    def __init__(self, input_dim, hidden_dim,dropout):
        super(CNN_RUL_student, self).__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.dropout=dropout
        self.encoder = nn.Sequential(
            nn.Conv1d(self.input_dim, 32, kernel_size=3, stride=2, padding=1, dilation=1),
            nn.Tanh(),
            nn.Conv1d(32, 16, kernel_size=3, stride=2, padding=1, dilation=1),
            nn.Tanh(),
            nn.Conv1d(16, 16, kernel_size=3, stride=4, padding=1, dilation=1),
            nn.Tanh(),
            # nn.LeakyReLU(),
            # nn.Conv1d(8, 8, kernel_size=3, stride=2, padding=1, dilation=1),
            # nn.LeakyReLU(),
            # nn.Conv1d(8, 8, kernel_size=3, stride=2, padding=1, dilation=1),
            # nn.LeakyReLU(),
            # nn.Conv1d(8, 8, kernel_size=3, stride=2, padding=1, dilation=1),
            nn.Flatten(),
            # nn.Linear(64, self.hidden_dim)
            )
        # adapter
        # self.adapter = nn.Conv1d(self.hidden_dim,64,kernel_size=1,stride=2,padding=0,dilation=1)  # output = teacher network feature output
        self.adapter = nn.Linear(32, 64)  # output = teacher network feature output
        self.regressor= nn.Sequential(
            nn.Linear(32, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.hidden_dim, 1))

    def forward(self, src):
        features = self.encoder(src)
        # hint = self.adapter(features.unsqueeze_(2)).squeeze()
        # predictions = self.regressor(features.squeeze())
        hint = self.adapter(features)
        predictions = self.regressor(features)
        return predictions.squeeze(), features, hint


class CNN_RUL_student_stack(nn.Module):
    def __init__(self, input_dim, hidden_dim,dropout):
        super(CNN_RUL_student_stack, self).__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.dropout=dropout

        self.encoder_1 = nn.Sequential(
            nn.Conv1d(self.input_dim, self.input_dim, kernel_size=3, stride=2, padding=1, dilation=4),
            # nn.Tanh(),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=12)
        )
        self.encoder_2 = nn.Sequential(
            nn.Conv1d(self.input_dim, self.input_dim, kernel_size=5, stride=2, padding=1, dilation=2),
            # nn.Tanh(),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=12)
        )
        self.encoder_3 = nn.Sequential(
            nn.Conv1d(self.input_dim, self.input_dim, kernel_size=7, stride=2, padding=1, dilation=1),
            # nn.Tanh(),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=13)
        )

        self.flat = nn.Flatten()
        # adapter
        # self.adapter = nn.Conv1d(42,64,kernel_size=1,stride=2,padding=0,dilation=1)  # output = teacher network feature output
        self.adapter = nn.Linear(42, 64)  # output = teacher network feature output
        self.regressor= nn.Sequential(
            nn.Linear(42, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.hidden_dim, 1)
        )

    def forward(self, src):
        fea_1 = self.encoder_1(src)
        fea_2 = self.encoder_2(src)
        fea_3 = self.encoder_3(src)
        features = self.flat(torch.cat((fea_1,fea_2,fea_3),dim=2))
        # features = self.encoder(src)
        # hint = self.adapter(features.unsqueeze_(2)).squeeze()
        # predictions = self.regressor(features.squeeze())
        hint = self.adapter(features)
        predictions = self.regressor(features)
        return predictions.squeeze(), features, hint


class CNN_RUL_student_stack_2(nn.Module):
    def __init__(self, input_dim, hidden_dim,dropout):
        super(CNN_RUL_student_stack_2, self).__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.dropout=dropout

        self.encoder_1 = nn.Sequential(
            nn.Conv1d(self.input_dim, self.input_dim, kernel_size=3, stride=2, padding=1, dilation=1),
            nn.LeakyReLU(),
            # nn.Tanh(),
            nn.Conv1d(self.input_dim,self.input_dim,kernel_size=3,stride=2,padding=1,dilation=4),
            nn.LeakyReLU(),
            # nn.Tanh(),
            nn.MaxPool1d(kernel_size=5)
        )
        self.encoder_2 = nn.Sequential(
            nn.Conv1d(self.input_dim, self.input_dim, kernel_size=3, stride=2, padding=1, dilation=1),
            nn.LeakyReLU(),
            # nn.Tanh(),
            nn.Conv1d(self.input_dim, self.input_dim, kernel_size=5, stride=2, padding=1, dilation=2),
            nn.LeakyReLU(),
            # nn.Tanh(),
            nn.MaxPool1d(kernel_size=5)
        )
        self.encoder_3 = nn.Sequential(
            nn.Conv1d(self.input_dim, self.input_dim, kernel_size=3, stride=2, padding=1, dilation=1),
            nn.LeakyReLU(),
            # nn.Tanh(),
            nn.Conv1d(self.input_dim, self.input_dim, kernel_size=7, stride=2, padding=1, dilation=1),
            nn.LeakyReLU(),
            # nn.Tanh(),
            nn.MaxPool1d(kernel_size=6)
        )
        self.flat = nn.Flatten()
        # adapter
        # self.adapter = nn.Conv1d(42,64,kernel_size=1,stride=2,padding=0,dilation=1)  # output = teacher network feature output
        self.adapter = nn.Linear(42, 64)  # output = teacher network feature output
        self.regressor= nn.Sequential(
            nn.Linear(64, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.hidden_dim, 1)
        )

    def forward(self, src):
        fea_1 = self.encoder_1(src)
        fea_2 = self.encoder_2(src)
        fea_3 = self.encoder_3(src)
        features = self.flat(torch.cat((fea_1,fea_2,fea_3),dim=2))
        # features = self.encoder(src)
        # hint = self.adapter(features.unsqueeze_(2)).squeeze()
        # predictions = self.regressor(features.squeeze())
        hint = self.adapter(features)
        # predictions = self.regressor(features)
        predictions = self.regressor(hint)
        return predictions.squeeze(), features, hint


class CNN_RUL_student_stack_test(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout):
        super(CNN_RUL_student_stack_test, self).__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.dropout = dropout

        self.encoder_1 = nn.Sequential(
            nn.Conv1d(self.input_dim, self.input_dim, kernel_size=5, stride=2, padding=1, dilation=1),
            nn.LeakyReLU(),
            # nn.Tanh(),
            nn.Conv1d(self.input_dim, self.input_dim, kernel_size=3, stride=2, padding=1, dilation=4),
            nn.LeakyReLU(),
            # nn.Tanh(),
            nn.MaxPool1d(kernel_size=4)
        )
        self.encoder_2 = nn.Sequential(
            nn.Conv1d(self.input_dim, self.input_dim, kernel_size=5, stride=2, padding=1, dilation=1),
            nn.LeakyReLU(),
            # nn.Tanh(),
            nn.Conv1d(self.input_dim, self.input_dim, kernel_size=5, stride=2, padding=1, dilation=2),
            nn.LeakyReLU(),
            # nn.Tanh(),
            nn.MaxPool1d(kernel_size=4)
        )
        self.encoder_3 = nn.Sequential(
            nn.Conv1d(self.input_dim, self.input_dim, kernel_size=5, stride=2, padding=1, dilation=1),
            nn.LeakyReLU(),
            # nn.Tanh(),
            nn.Conv1d(self.input_dim, self.input_dim, kernel_size=7, stride=2, padding=1, dilation=1),
            nn.LeakyReLU(),
            # nn.Tanh(),
            nn.MaxPool1d(kernel_size=5)
        )
        self.flat = nn.Flatten()
        # adapter
        # self.adapter = nn.Conv1d(42,64,kernel_size=1,stride=2,padding=0,dilation=1)  # output = teacher network feature output
        self.adapter = nn.Linear(42, 64)  # output = teacher network feature output
        self.regressor = nn.Sequential(
            nn.Linear(42, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.hidden_dim, 1)
        )

    def forward(self, src):
        fea_1 = self.encoder_1(src)
        fea_2 = self.encoder_2(src)
        fea_3 = self.encoder_3(src)
        features = self.flat(torch.cat((fea_1, fea_2, fea_3), dim=2))
        # features = self.encoder(src)
        # hint = self.adapter(features.unsqueeze_(2)).squeeze()
        # predictions = self.regressor(features.squeeze())
        hint = self.adapter(features)
        predictions = self.regressor(features)
        return predictions.squeeze(), features, hint


class CNN_RUL_student_glu(nn.Module):
    def __init__(self, input_dim, hidden_dim,dropout):
        super(CNN_RUL_student_glu, self).__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.dropout=dropout

        self.encoder_1 = nn.Sequential(
            nn.Conv1d(self.input_dim, 8, kernel_size=3, stride=2, padding=1, dilation=1),
            nn.LeakyReLU(),
        )
        # self.encoder_2 = nn.Sequential(
        #     nn.Conv1d(4, 8, kernel_size=3, stride=2, padding=1, dilation=1),
        #     nn.LeakyReLU(),
        # )
        self.flat = nn.Flatten()
        # adapter
        # self.adapter = nn.Conv1d(4,64,kernel_size=1,stride=2,padding=0,dilation=1)  # output = teacher network feature output
        self.adapter = nn.Linear(60, 64)  # output = teacher network feature output
        self.regressor= nn.Sequential(
            nn.Linear(60, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.hidden_dim, 1)
        )

    def forward(self, src):
        fea_1 = self.encoder_1(src)
        fea_1 = F.glu(fea_1,dim=1)
        # fea_1 = self.encoder_2(fea_1)
        # fea_1 =F.glu(fea_1,dim=1)

        features = self.flat(fea_1)
        # features = self.encoder(src)
        # features = fea_1
        # hint = self.adapter(features.unsqueeze_(2)).squeeze()
        # predictions = self.regressor(features.squeeze())
        hint = self.adapter(features)
        predictions = self.regressor(features)
        return predictions.squeeze(), features, hint


class MyDataset(Dataset):
    def __init__(self, x, y):
        super(MyDataset, self).__init__()
        self.len = x.shape[0]
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
        self.x_data = torch.as_tensor(x, device=device, dtype=torch.float)
        self.y_data = torch.as_tensor(y, device=device, dtype=torch.float)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


def RMSELoss(yhat,y):
    return torch.sqrt(torch.mean((yhat-y)**2))


def Score(yhat,y):
    score =0
    for i in range(len(yhat)):
        if y[i] <= yhat[i]:
            score = score + torch.exp(-(y[i]-yhat[i])/10.0)-1
        else:
            score = score + torch.exp((y[i]-yhat[i])/13.0)-1
    return score.item()


def centered_average(nums):
    return sum(nums) / len(nums)


def weights_init(m):
    for child in m.children():
        if isinstance(child,nn.Linear) or isinstance(child,nn.Conv1d):
            torch.nn.init.xavier_uniform_(child.weight)


class Generator(nn.Module):
    def __init__(self, input_dim):
        super(Generator, self).__init__()
        self.input_dim = input_dim
        self.encoder = nn.Sequential(
            nn.Conv1d(self.input_dim, 8, kernel_size=4, stride=1, padding=1),
            nn.Sigmoid(),
            nn.AvgPool1d(kernel_size=2), # [?,8,14]

            nn.Conv1d(8, 14, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
            nn.AvgPool1d(kernel_size=2), #[?,14,7]
        )
        self.flat = nn.Flatten()
        self.adapter = nn.Linear(98, 64)  # output = teacher network feature output

    def forward(self, src):
        fea= self.encoder(src)
        fea = self.flat(fea)
        hint = self.adapter(fea)
        return fea, hint


class CNN_RUL_student_conventional(nn.Module):
    # Model implementation of Paper Deep Convolutional Neural Network Based Regression Approach  for Estimation of RUL
    # Difference: The input dim is different, ours [?,14,30], paper[?, 15, 27]
    def __init__(self, input_dim):
        super(CNN_RUL_student_conventional, self).__init__()
        self.input_dim = input_dim
        self.generator = Generator(input_dim=input_dim)
        self.regressor = nn.Linear(98, 1)

    def forward(self, src):
        features,hint = self.generator(src)
        predictions = self.regressor(features)
        return predictions.squeeze(), features, hint
