#!/usr/bin/env python
#-*- coding: utf-8 -*-
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from models import LSTM_RUL, RMSELoss, Score, centered_average, MyDataset,weights_init
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import argparse
import logging


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


class Discriminator(nn.Module):
    def __init__(self, input_features):
        super(Discriminator, self).__init__()
        self.input_features = input_features

        # Define hidden linear layers
        self.fc1 = nn.Linear(input_features,input_features//2)
        self.fc2 = nn.Linear(input_features//2, input_features//4)
        self.out = nn.Linear(input_features//4,1)
        # Activation function
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):
        x = self.leaky_relu(self.fc1(x))
        x = self.leaky_relu(self.fc2(x))
        x = torch.sigmoid(self.out(x))
        return x


class Generator_2(nn.Module):
    def __init__(self, input_dim):
        super(Generator_2, self).__init__()
        self.input_dim = input_dim
        self.encoder_1 = nn.Sequential(
            nn.Conv1d(self.input_dim, self.input_dim, kernel_size=3, stride=2, padding=1, dilation=1),
            nn.LeakyReLU(),
            # nn.Tanh(),
            nn.Conv1d(self.input_dim, self.input_dim, kernel_size=3, stride=2, padding=1, dilation=4),
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
        self.adapter = nn.Linear(42, 64)  # output = teacher network feature output

    def forward(self, src):
        fea_1 = self.encoder_1(src)
        fea_2 = self.encoder_2(src)
        fea_3 = self.encoder_3(src)
        features = self.flat(torch.cat((fea_1,fea_2,fea_3),dim=2))
        hint = self.adapter(features)
        return features,hint


class CNN_RUL_student_stack_2(nn.Module):
    def __init__(self, input_dim, hidden_dim,dropout):
        super(CNN_RUL_student_stack_2, self).__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.dropout=dropout
        self.generator = Generator_2(input_dim=input_dim)
        self.regressor= nn.Sequential(
            nn.Linear(64, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.hidden_dim, 1)
        )

    def forward(self, src):
        features,hint = self.generator(src)
        predictions = self.regressor(hint)
        # predictions = self.regressor(features)
        return predictions.squeeze(), features, hint


def joint_loss_combined_2(yhat_s, yhat_t, ytrue, alpha=0.0):
    yhat_t = Variable(yhat_t,requires_grad=False)
    hard_loss = nn.MSELoss()(yhat_s,ytrue)
    soft_loss = nn.MSELoss()(yhat_s,yhat_t)
    loss = (1-alpha) * hard_loss + alpha * soft_loss
    return loss


def hint_loss_L2(feature_s,feature_t):
    # hint_loss = nn.MSELoss()(feature_s,feature_t)
    hint_loss = nn.L1Loss()(feature_s,feature_t)
    # hint_loss = nn.SmoothL1Loss()(feature_s,feature_t)
    return hint_loss


def train_hint(model_s, model_t, device,train_loader,optimizer,step):
    model_t.eval()
    Loss = 0.0
    for batch_idx, (batch_x,batch_y) in enumerate(train_loader):
        batch_x,batch_y = batch_x.to(device),batch_y.to(device)
        optimizer.zero_grad()
        # Forward
        _, _, hint = model_s(batch_x.permute(0,2,1))
        if step==1:
            _,feature_t = model_t(batch_x)
        else:
            _,_,feature_t=model_t(batch_x.permute(0,2,1))

        feature_t = Variable(feature_t, requires_grad=False)
        # Calculate loss and update weights
        loss = hint_loss_L2(hint,feature_t)
        # log_hint = F.log_softmax(hint)
        # loss = hint_loss_KLD(log_hint,feature_t)
        # loss = hint_loss_log_cosh(hint,feature_t)
        # loss =hint_loss_RMSLE(hint,feature_t)
        # loss = HoMM3(hint,feature_t)
        # loss = HoMM4(hint,feature_t)
        # loss = KMMD(hint,feature_t,device)
        # loss =KernelHoMM(hint,feature_t,sigma=0.01,device=device)
        # loss = KHoMM(hint,feature_t,sigma=0.01,device=device,num=256,order=2)
        # loss = hint_loss_mmd(hint,feature_t)
        loss.backward()
        optimizer.step()
        Loss += loss.item()
    return Loss/len(train_loader)


def train_kd(model_s, model_t, device,train_loader,optimizer,alpha,step):
    model_t.eval()
    Loss = 0.0
    for batch_idx, (batch_x,batch_y) in enumerate(train_loader):
        batch_x,batch_y = batch_x.to(device),batch_y.to(device)
        optimizer.zero_grad()
        # Forward
        pred_s, _, _= model_s(batch_x.permute(0,2,1))
        if step==1:
            pred_t, _ = model_t(batch_x)
        else:
            pred_t, _, _ = model_t(batch_x.permute(0, 2, 1))
        # Calculate loss and update weights
        loss = joint_loss_combined_2(pred_s, pred_t, batch_y, alpha)
        loss.backward()
        optimizer.step()
        Loss += loss.item()
    return Loss/len(train_loader)


def test(model,device,x,y,max_RUL):
    model.eval()
    with torch.no_grad():
        x_cuda,y_cuda = x.to(device),y.to(device)
        pred, _, _= model(x_cuda.permute(0,2,1))
        pred = pred *max_RUL
        rmse = RMSELoss(pred,y_cuda)
        score = Score(pred,y_cuda)
        return rmse, score
    pass


def validate(model,device,x,y):
    model.eval()
    with torch.no_grad():
        x_cuda,y_cuda = x.to(device),y.to(device)
        pred, _, _ = model(x_cuda.permute(0,2,1))
        loss = RMSELoss(pred,y_cuda)
    return loss


def train_gan(model_s, model_t, netD, train_loader, device,criterion, optimizerD, optimizerG, epoch, hint_epochs):

    # Establish convention for real and fake labels during training
    real_label = 1
    fake_label = 0

    # For each batch in dataloader
    for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        ########################################################
        # (1) update D network: maximize log(D(fea_t)) + log(1-D(G(x))
        # fea_t: feature from teacher network
        # x: input data
        # G(x): feature from student network
        ########################################################
        ## Train with all-real batch
        netD.zero_grad()

        # Format Batch
        _, feature_t = model_t(batch_x)
        feature_t = Variable(feature_t, requires_grad=False)
        label = torch.full((batch_x.shape[0],), real_label, dtype=torch.float, device=device)
        # Forward pass real batch through D
        output = netD(feature_t).view(-1)
        # Calculate loss on all-real batch
        errD_real = criterion(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        # noise = torch.randn(batch_x.shape).to(device)
        # Generate fake features with G
        _, _, fake = model_s(batch_x.permute(0, 2, 1))
        label.fill_(fake_label)
        # Classify all fake batch with D
        output = netD(fake.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, label)
        # Calculate the gradients for this batch
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Add the gradients from all-real and all-fake batches
        errD = errD_real + errD_fake
        # Update D
        optimizerD.step()

        ########################################################
        # (2) update G network: maximize log(D(G(x))
        # x: input data
        ########################################################
        model_s.generator.zero_grad()
        # fake labels are real for generator cost
        label.fill_(real_label)
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = netD(fake).view(-1)
        # Calculate G's loss based on this output
        errG = criterion(output, label)
        # Calculate gradients for G
        # errG.backward()
        D_G_z2 = output.mean().item()

        # Add L1 loss
        errL1 = nn.L1Loss()(fake, feature_t)
        errG_t = errG + errL1

        # errG_t = errG
        errG_t.backward()

        # Update G
        optimizerG.step()

        if epoch % 100 == 0:
            print(
                '[%d/%d][%2d/%d]\tLoss_D_real:%.4f\tLoss_D_fake:%.4f\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                % (epoch, hint_epochs, batch_idx, len(train_loader), errD_real.item(), errD_fake.item(), errD.item(),
                   errG.item(), D_x, D_G_z1, D_G_z2))

    return errG, errD, errD_real,errD_fake, errL1

    pass


def main(args):
    train_enable = (args.train == 0)
    dataset_index = args.dataset
    data_identifiers = ['FD001', 'FD002', 'FD003', 'FD004']
    data_identifier = data_identifiers[dataset_index - 1]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    max_RUL= 130
    kd_epochs = 40
    hint_epochs = 40
    gan_epochs = 80
    batch_size = 256
    generations = 5
    iterations = 1
    lrate = 1e-3
    lr_gan = 1e-4
    beta1 = 0.5
    alphas = [0.7]


    model_path = 'result/'
    data_path = 'processed_data/cmapss_train_valid_test_dic.pt'
    my_dataset = torch.load(data_path)

    if not os.path.exists('./log/'):
        os.makedirs('./log/')

    logger_name = 'log/' + data_identifier + '_log_gan_seq.log'
    logging.basicConfig(filename=logger_name, level=logging.INFO)

    # Create Training Dataset
    train_x = torch.from_numpy(my_dataset[data_identifier]['train_data']).float()
    train_y = torch.from_numpy(my_dataset[data_identifier]['train_labels']).float()
    test_x = torch.from_numpy(my_dataset[data_identifier]['test_data']).float()
    test_y = torch.FloatTensor(my_dataset[data_identifier]['test_labels'])
    val_x = torch.from_numpy(my_dataset[data_identifier]['valid_data']).float()
    val_y = torch.from_numpy(my_dataset[data_identifier]['valid_labels']).float()
    train_ds = MyDataset(train_x,train_y)
    train_loader = DataLoader(dataset=train_ds,batch_size=batch_size,shuffle=True,num_workers=0,worker_init_fn=_init_fn)

    test_rmse_alpha = []
    test_score_alpha = []

    for alpha in alphas:
        test_rmse = []
        test_score = []
        for iteration in range(iterations):
            for gen in range(1,generations+1):
                best_val_loss = 1e+9
                if gen == 1:
                    ## For the first Iteration 0: train the student with LSTM teacher
                    model_t = LSTM_RUL(input_dim=14, hidden_dim=32, n_layers=5, dropout=0.5, bid=True, device=device).to(device)
                    model_name = model_path + data_identifier + '_teacher.pt'
                    model_t.load_state_dict(torch.load(model_name))
                    # Get the teacher pretrained layer parameters
                    pretrained_dcit = model_t.state_dict()
                else:
                    ## For the k iteration(k>1), train a new student with the k-1 saved student model
                    model_t = CNN_RUL_student_stack_2(input_dim=14, hidden_dim=32, dropout=0.5).to(device)
                    model_name = model_path + data_identifier + '_student_seq_gan_'+str(gen-1)+'_alpha_'+str(alpha)+'.pt'
                    model_t.load_state_dict(torch.load(model_name))

                #Create a new student
                print("Start training dataset", data_identifier, "alpha = ", alpha)
                logging.info("Start training dataset {} alpah ={}".format(data_identifier, alpha))
                model_s = CNN_RUL_student_stack_2(input_dim=14, hidden_dim=32, dropout=0.5).to(device)
                model_s.apply(weights_init)

                if gen == 1:
                    # For the First Generation: use GAN method to train new student

                    # Update Student regressor parameters with teacher pretrained parameters
                    # model_dict = model_s.state_dict()
                    # pretrained_dcit = {k: v for k, v in pretrained_dcit.items() if k in model_dict}
                    # model_dict.update(pretrained_dcit)
                    # model_s.load_state_dict(model_dict)

                    # Create Discriminator
                    netD = Discriminator(input_features=64).to(device)

                    # Initialize BCELoss function
                    criterion = nn.BCELoss()

                    # Setup Adam optimizer for both G and D
                    optimizerD = optim.Adam(netD.parameters(), lr=lr_gan, betas=(beta1, 0.999))
                    optimizerG = optim.Adam(model_s.generator.parameters(), lr=lr_gan, betas=(beta1, 0.999))

                    G_losses = []
                    G_l1_losses = []
                    D_losses = []
                    D_losses_real = []
                    D_losses_fake = []

                    print("Starting GAN Training loop...")
                    logging.info("Starting GAN Training loop...")
                    # For each epoch
                    for epoch in range(gan_epochs):
                        errG, errD, errD_real, errD_fake, errL1 = train_gan(model_s, model_t, netD, train_loader, device,
                                                                            criterion, optimizerD, optimizerG, epoch,
                                                                            gan_epochs)
                        # Save Losses for plotting later
                        G_losses.append(errG.item())
                        D_losses.append(errD.item())
                        D_losses_real.append(errD_real.item())
                        D_losses_fake.append(errD_fake.item())
                        G_l1_losses.append((errL1.item()))
                else:
                    # For the rest Generation: use Hint method to train new student
                    optimizer =optim.AdamW(model_s.parameters(), lr=0.001)
                    scheduler = StepLR(optimizer, gamma=0.9, step_size=10)
                    # Hint Learning
                    for epoch in range(hint_epochs):
                        epoch_loss = train_hint(model_s,model_t,device,train_loader,optimizer,gen)
                        if epoch % 10 ==0:
                            print('Iteration-{} Generation-{} Epoch-{} Hint Loss= {:.9f}'.format(iteration, gen, epoch, epoch_loss))
                        scheduler.step()

                optimizer = optim.AdamW(model_s.parameters(), lr=lrate)
                scheduler = StepLR(optimizer, gamma=0.9, step_size=10)
                # KD Learning
                for epoch in range(kd_epochs):
                    epoch_loss = train_kd(model_s,model_t,device,train_loader,optimizer,alpha,gen)
                    scheduler.step()
                    if epoch % 10 == 0:
                        print('Iteration-{} Generation-{} Epoch{} Training Loss= {:.9f}'.format(iteration, gen, epoch, epoch_loss))
                        with torch.no_grad():
                            val_loss = validate(model_s,device,val_x,val_y)
                            if val_loss < best_val_loss:
                                print("val_loss improved from {:.9f} to {:.9f}, save the model".format(best_val_loss,val_loss))
                                best_val_loss = val_loss
                                model_s_best = model_path + data_identifier + '_student_seq_gan_' + str(gen) + '_alpha_' + str(
                                    alpha) + '.pt'
                                torch.save(model_s.state_dict(), model_s_best)
                # Evaluate on test data set
                # Load the best student
                model_s.load_state_dict(torch.load(model_s_best))
                rmse,score = test(model_s,device,test_x,test_y,max_RUL)
                print("Iteration-{} Generation-{} Test RMSE = {:.9f}, Test Score ={:.9f}".format(iteration, gen,rmse,score))
                logging.info("Iteration-{} Generation-{} Test RMSE = {:.2f}, Test Score ={:.2f}".
                             format(iteration, gen, rmse, score))

            # append the last generation rmse and score
            test_rmse.append(rmse.item())
            test_score.append(score)
        print("{} Alpha={} | Test RMSE={}".format(data_identifier, alpha, test_rmse))
        print("{} Alpha={} | Test Score={}".format(data_identifier, alpha, test_score))
        logging.info("{} Alpha={} | Test RMSE={}".format(data_identifier, alpha, test_rmse))
        logging.info("{} Alpha={} | Test Score={}".format(data_identifier, alpha, test_score))
        mean_rmse = centered_average(test_rmse)
        mean_score = centered_average(test_score)
        print("{} Alpha={} | Average RMSE = {}".format(data_identifier, alpha, mean_rmse))
        print("{} Alpha={} | Average Score = {}".format(data_identifier, alpha, mean_score))
        logging.info("{} Alpha={} | Average RMSE = {}".format(data_identifier, alpha, mean_rmse))
        logging.info("{} Alpha={} | Average Score = {}".format(data_identifier, alpha, mean_score))
        test_rmse_alpha.append(mean_rmse)
        test_score_alpha.append(mean_score)
    print("{} | Whole RMSE = {}".format(data_identifier, test_rmse_alpha))
    print("{} | Whole Score = {}".format(data_identifier, test_score_alpha))
    logging.info("{} | Whole RMSE = {}".format(data_identifier, test_rmse_alpha))
    logging.info("{} | Whole Score = {}".format(data_identifier, test_score_alpha))


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
