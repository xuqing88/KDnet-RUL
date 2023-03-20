import os
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from models import LSTM_RUL, RMSELoss, Score, centered_average, MyDataset,LSTM_RUL_Student_simple
from torch.optim.lr_scheduler import StepLR
import time
import argparse

def _init_fn(worker_id):
    np.random.seed(int(1))
def set_random_seed(seed):
    np.random.seed(seed)
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
set_random_seed(4)

def train(model,device,train_loader,optimizer):
    model.train()
    Loss = 0.0
    critirion = nn.MSELoss()
    epoch_train_time =0.0
    cnt=0
    for batch_idx, (batch_x,batch_y) in enumerate(train_loader):
        t0=time.perf_counter()

        batch_x,batch_y = batch_x.to(device),batch_y.to(device)
        optimizer.zero_grad()
        # Forward
        pred, _ = model(batch_x)
        # Calculate loss and update weights
        loss = critirion(pred,batch_y)
        loss.backward()
        optimizer.step()
        Loss += loss.item()

        t1=time.perf_counter()
        epoch_train_time+= (t1-t0)
        # print("Batch training time={}".format(t1-t0))
        cnt += 1
    # print("Average Batch Training Time=",epoch_train_time/cnt)
    return Loss/len(train_loader)


def test(model,device,x,y,max_RUL):
    model.eval()
    with torch.no_grad():
        x_cuda,y_cuda = x.to(device),y.to(device)
        pred, _ = model(x_cuda)
        pred = pred *max_RUL
        rmse = RMSELoss(pred,y_cuda)
        score = Score(pred,y_cuda)
        return rmse, score
    pass


def main(args):
    train_enable = (args.train==0)
    dataset_index = args.dataset
    data_identifiers = ['FD001', 'FD002', 'FD003', 'FD004']
    data_identifier = data_identifiers[dataset_index - 1]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    max_RUL = 130
    epochs = 40
    batch_size = 64
    lrate = 1e-3
    iterations = 1

    if not os.path.exists('./result/'):
        os.makedirs('./result/')
    model_path= 'result/'
    data_path = 'processed_data/cmapss_train_valid_test_dic.pt'
    my_dataset = torch.load(data_path)
    model_name = model_path + data_identifier + '_teacher.pt'

    # Create Training Dataset
    train_x = torch.from_numpy(my_dataset[data_identifier]['train_data']).float()
    train_y = torch.from_numpy(my_dataset[data_identifier]['train_labels']).float()
    test_x = torch.from_numpy(my_dataset[data_identifier]['test_data']).float()
    test_y = torch.FloatTensor(my_dataset[data_identifier]['test_labels'])
    train_ds = MyDataset(train_x,train_y)
    train_loader = DataLoader(dataset=train_ds,batch_size=batch_size,shuffle=True,num_workers=0,worker_init_fn=_init_fn)
    test_rmse = []
    test_score = []
    total_time = 0.0
    for iteration in range(iterations):
        model = LSTM_RUL(input_dim=14, hidden_dim=32, n_layers=5, dropout=0.5, bid=True, device=device).to(device)
        if train_enable:
            print("Start training dataset", data_identifier)
            optimizer =optim.AdamW(model.parameters(), lr=lrate)
            scheduler = StepLR(optimizer,gamma=0.9,step_size=1)

            for epoch in range(epochs):
                epoch_loss = train(model,device,train_loader,optimizer)
                scheduler.step()
                print('Epoch{} Loss= {:.6f}'.format(epoch, epoch_loss))
            rmse,score = test(model,device,test_x,test_y,max_RUL)
            print("Test RMSE = {}, Test Score ={}".format(rmse,score))
            test_rmse.append(rmse.item())
            test_score.append(score)
            # save trained model
            torch.save(model.state_dict(),model_name)
        else:
            model_name = model_path + data_identifier + '_teacher.pt'
            t0 = time.perf_counter()
            model.load_state_dict(torch.load(model_name))
            rmse,score = test(model,device,test_x,test_y,max_RUL)
            t1 = time.perf_counter()
            print("Test RMSE = {}, Test Score ={}".format(rmse,score))
            print("Inference time = ",t1-t0)
            if iteration !=0:
                total_time+=(t1-t0)
        print("{} Test RMSE={}".format(data_identifier, test_rmse))
        print("{} Test Score={}".format(data_identifier, test_score))
    if not train_enable:
        print("Average Training Time={}".format(total_time/(iterations-1)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train',
                        help='0:Model Train, 1: Model Inference',
                        type=int,
                        choices=[0, 1],
                        default=0)
    parser.add_argument('-d', '--dataset',
                        help='1:FD001, 2: FD002, 3:FD003, 4:FD004',
                        type=int,
                        choices=[1, 2, 3, 4],
                        default=1)
    args = parser.parse_args()
    main(args)
