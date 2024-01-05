import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, MinMaxScaler

class LSTM(nn.Module):
    
    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        super().__init__()
        self.num_classes = num_classes # output size
        self.num_layers = num_layers # number of recurrent layers in the lstm
        self.input_size = input_size # input size
        self.hidden_size = hidden_size # neurons in each lstm layer
        # LSTM model
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, dropout=0.2) # lstm
        self.fc_1 =  nn.Linear(hidden_size, 128) # fully connected 
        self.fc_2 = nn.Linear(128, num_classes) # fully connected last layer
        self.relu = nn.ReLU()
        
    def forward(self,x):
        # hidden state
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        # cell state
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        # propagate input through LSTM
        output, (hn, cn) = self.lstm(x, (h_0, c_0)) # (input, hidden, and internal state)
        hn = hn.view(-1, self.hidden_size) # reshaping the data for Dense layer next
        out = self.relu(hn)
        out = self.fc_1(out) # first dense
        out = self.relu(out) # relu
        out = self.fc_2(out) # final output
        return out

def training_loop(n_epochs, lstm, optimiser, loss_fn, X_train, y_train,
                  X_test, y_test):
    test_loss = 0
    for epoch in range(n_epochs):
        lstm.train()
        outputs = lstm.forward(X_train) # forward pass
        optimiser.zero_grad() # calculate the gradient, manually setting to 0
        # obtain the loss function
        loss = loss_fn(outputs, y_train)
        loss.backward() # calculates the loss of the loss function
        optimiser.step() # improve from loss, i.e backprop
        # test loss
        lstm.eval()
        test_preds = lstm(X_test)
        test_loss = loss_fn(test_preds, y_test)
        if epoch % 100 == 0:
            print("Epoch: %d, train loss: %1.5f, test loss: %1.5f" % (epoch, 
                                                                      loss.item(), 
                                                                      test_loss.item()))
    return test_loss.item()

# not used anymore
def test_stationarity(timeseries):
    '''
    Input: timeseries (dataframe): timeseries for which we want to study the stationarity
    '''
    
    #Determing rolling statistics
    rolmean = timeseries.rolling(20).mean()
    rolstd = timeseries.rolling(20).std()

    #Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)

    #Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value',\
                                             '#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)

n_epochs = 500 
learning_rate = 0.001 

preview = 241 # How many previous days to use for prediction
predict = 1 # how many values in the future to predict
futureStep = 241 # how far in the future to predict

total = preview+predict

hidden_size = 10 # number of features in hidden state
num_layers = 1 # number of stacked lstm layers

num_classes = predict # number of output classes 

ESG = False
input_size = 6 # number of features (without ESG 6 with 19)
if ESG:
    input_size = 19

lstm = LSTM(num_classes, 
        input_size, 
        hidden_size, 
        num_layers)
loss_fn = torch.nn.MSELoss()    # mean-squared error for regression
optimiser = torch.optim.Adam(lstm.parameters(), lr=learning_rate)

df2019 = pd.read_excel('data_feature_csi_2019.xlsx', index_col = 'Company_code', parse_dates=True)
df2020 = pd.read_excel('data_feature_csi_2020.xlsx', index_col = 'Company_code', parse_dates=True)
df2021 = pd.read_excel('data_feature_csi_2021.xlsx', index_col = 'Company_code', parse_dates=True)
df2022 = pd.read_excel('data_feature_csi_2022.xlsx', index_col = 'Company_code', parse_dates=True)

def TrainingIteration(plotting, f, filename):

    df = pd.read_csv(f, index_col = 'Date', parse_dates=True)

    #Remove years without ESG Data
    df = df[~(df.index.year == 2018)]
    df = df[~(df.index.year == 2019)]
    X, y = df.drop(columns=['Close']), df.Close.values

    # Add ESG Data --------------------------------------

    #Get Amount of data per year
    X.loc[ X.index.year == 2018].sum(axis=1).count()
    X.loc[ X.index.year == 2019].sum(axis=1).count()
    count2020 = X.loc[ X.index.year == 2020].sum(axis=1).count()
    count2021 =X.loc[ X.index.year == 2021].sum(axis=1).count()
    count2022 =X.loc[ X.index.year == 2022].sum(axis=1).count()
    count2023 =X.loc[ X.index.year == 2023].sum(axis=1).count()

    # create Stacks of ESG values to be added to training data.

    companyID = filename.split('_',1)[0]
    print(companyID)

    ESG2019 = df2019.loc[df2019.index == companyID]
    dub = np.tile(ESG2019.values, (count2020, 1))
    ESG2019_df = pd.DataFrame(dub, columns=ESG2019.columns)

    ESG2020 = df2020.loc[df2020.index == companyID]
    dub = np.tile(ESG2020.values, (count2021, 1))
    ESG2020_df = pd.DataFrame(dub, columns=ESG2020.columns)

    ESG2021 = df2021.loc[df2021.index == companyID]
    dub = np.tile(ESG2021.values, (count2022, 1))
    ESG2021_df = pd.DataFrame(dub, columns=ESG2021.columns)

    ESG2022 = df2022.loc[df2022.index == companyID]
    dub = np.tile(ESG2022.values, (count2023, 1))
    ESG2022_df = pd.DataFrame(dub, columns=ESG2022.columns)


    ESGData = pd.concat([ESG2019_df, ESG2020_df,ESG2021_df,ESG2022_df], ignore_index=True)
    
    ESGData= ESGData.reset_index(drop=True)
    X = X.reset_index(drop=True)
    
    X_ESG = pd.concat([X, ESGData], axis=1)
    print(X.shape)
    print(ESGData.shape)
    print(X_ESG.shape)

    X_ESG = X_ESG.drop('Industry_code', axis=1)
    X_ESG = X_ESG.fillna(0)
    # Add ESG Data --------------------------------------

    if ESG:
        X = X_ESG

    # Data Preprocessing ----------------------------------------------------------------
    mm = MinMaxScaler()
    ss = StandardScaler()

    #extra step to remove really all strings
    X = X.applymap(lambda x: pd.to_numeric(x, errors='coerce'))
    X = X.fillna(0)

    # Basic Scaling of input 
    X_trans = ss.fit_transform(X)
    y_trans = mm.fit_transform(y.reshape(-1, 1))
    
    # spliting the trainings data into sequences for training
    def split_sequences(input_sequences, output_sequence, n_steps_in, n_steps_out):
        X, y = list(), list() # instantiate X and y
        for i in range(len(input_sequences)):
            # find the end of the input, output sequence
            end_ix = i + n_steps_in
            out_end_ix = end_ix + n_steps_out - 1
            # check if we are beyond the dataset
            if out_end_ix +futureStep > len(input_sequences): break
        
            # gather input and output of the pattern
            #seq_x, seq_y = input_sequences[i:end_ix], output_sequence[end_ix-1:out_end_ix, -1]
            seq_x, seq_y = input_sequences[i:end_ix], output_sequence[futureStep+end_ix-1:futureStep+out_end_ix, -1]
            X.append(seq_x), y.append(seq_y)
        return np.array(X), np.array(y)

    X_ss, y_mm = split_sequences(X_trans, y_trans, preview, predict)

    print(X_ss.shape, y_mm.shape)

    total_samples = len(X)
    train_test_cutoff = total_samples-total #round(0.3 * total_samples)

    #split between train and testing sets
    testSamplesCount = round(0.3 * total_samples)

    X_train = X_ss[:-testSamplesCount]
    X_test = X_ss[-testSamplesCount:]

    y_train = y_mm[:-testSamplesCount]
    y_test = y_mm[-testSamplesCount:] 

    #print("Training Shape:", X_train.shape, y_train.shape)
    #print("Testing Shape:", X_test.shape, y_test.shape) 

    X_train_tensors = Variable(torch.Tensor(X_train))
    X_test_tensors = Variable(torch.Tensor(X_test))

    y_train_tensors = Variable(torch.Tensor(y_train))
    y_test_tensors = Variable(torch.Tensor(y_test))

    try:
        X_train_tensors_final = torch.reshape(X_train_tensors,   
                                    (X_train_tensors.shape[0], preview, 
                                    X_train_tensors.shape[2]))
        X_test_tensors_final = torch.reshape(X_test_tensors,  
                                            (X_test_tensors.shape[0], preview, 
                                            X_test_tensors.shape[2])) 
    except:
        print("skipped invalid data")
        return -1

    #print("Training Shape:", X_train_tensors_final.shape, y_train_tensors.shape)
    #print("Testing Shape:", X_test_tensors_final.shape, y_test_tensors.shape) 

    # Training ------------------------------------------------------------------------------

    finalTestLoss = training_loop(n_epochs=n_epochs,
        lstm=lstm,
        optimiser=optimiser,
        loss_fn=loss_fn,
        X_train=X_train_tensors_final,
        y_train=y_train_tensors,
        X_test=X_test_tensors_final,
        y_test=y_test_tensors)
    
    # Ploting ------------------------------------------------------------------------------
    
    test_predict = lstm(X_test_tensors_final[-1].unsqueeze(0)) # get the last sample
    test_predict = test_predict.detach().numpy()
    test_predict = mm.inverse_transform(test_predict)
    test_predict = test_predict[0].tolist()

    if plotting:
        """ 
        df_X_ss = ss.transform(X) # old transformers
        df_y_mm = mm.transform(df.Close.values.reshape(-1, 1)) # old transformers
        # split the sequence
        df_X_ss, df_y_mm = split_sequences(df_X_ss, df_y_mm, preview, predict)
        # converting to tensors
        df_X_ss = Variable(torch.Tensor(df_X_ss))
        df_y_mm = Variable(torch.Tensor(df_y_mm))
        # reshaping the dataset
        df_X_ss = torch.reshape(df_X_ss, (df_X_ss.shape[0], preview, df_X_ss.shape[2]))

        train_predict = lstm(df_X_ss) # forward pass
        data_predict = train_predict.data.numpy() # numpy conversion
        dataY_plot = df_y_mm.data.numpy()

        data_predict = mm.inverse_transform(data_predict) # reverse transformation
        dataY_plot = mm.inverse_transform(dataY_plot)
        true, preds = [], []
        for i in range(len(dataY_plot)):
            true.append(dataY_plot[i][0])
        for i in range(len(data_predict)):
            preds.append(data_predict[i][0])
        plt.figure(figsize=(10,6)) #plotting
        plt.axvline(x=train_test_cutoff, c='r', linestyle='--') # size of the training set

        plt.plot(true, label='Actual Data') # actual plot
        plt.plot(preds, label='Predicted Data') # predicted plot
        plt.title('Time-Series Prediction')
        plt.legend()
        plt.savefig("whole_plot.png", dpi=300)
        plt.show() 
        """

        # Ploting 2 ------------------------------------------------------------------------------

        test_target = y_test_tensors[-1].detach().numpy() # last sample again
        test_target = mm.inverse_transform(test_target.reshape(1, -1))
        test_target = test_target[0].tolist()

        plt.plot(test_target, label="Actual Data")
        plt.plot(test_predict, label="LSTM Predictions")
        plt.savefig("small_plot.png", dpi=300)
        plt.show()

        # Ploting 2 ------------------------------------------------------------------------------

        plt.figure(figsize=(10,6)) #plotting
        a = [x for x in range(0, len(y))]
        #a = [x for x in df.index]
        plt.plot(a, y[0:], label='Actual data')
        c = [x for x in range(len(y)-predict, len(y))]
        #plt.plot(c, test_predict, label=f'One-shot multi-step prediction ({predict} days)')
        #plt.axvline(x=len(y)-predict, c='r', linestyle='--')
        plt.axvline(x=train_test_cutoff, c='r', linestyle='--')

        plt.axvline(x=total_samples-count2023, c='b', linestyle='--')
        plt.axvline(x=total_samples-count2023-count2022, c='b', linestyle='--')
        plt.axvline(x=total_samples-count2023-count2022-count2021, c='b', linestyle='--')
        plt.axvline(x=total_samples-count2023-count2022-count2021-count2020, c='b', linestyle='--')
        plt.axvline(x=total_samples, c='b', linestyle='--')

        plt.scatter(len(y), test_predict[0], color='red', marker='x', label='LongTermPrediction')

        plt.legend()
        plt.show()

    return finalTestLoss


def main():
    safe = True
    iteration = 0
    finalloss = []
    for filename in os.listdir('CSI300_historical_Data'):
        iteration += 1
        print(f"iteration: {iteration}")
        f = os.path.join('CSI300_historical_Data', filename)
        print('read file:' + f)
        if iteration == 30:
            break
        
        loss = TrainingIteration(False, f, filename)
        if loss:
            finalloss.append(loss)

    print("Final Statistic")
    print(finalloss)
    print("Avarage Loss")
    print((sum(finalloss) / len(finalloss)))
    print("Finished")

    if safe:
        torch.save(lstm.state_dict(), "Model")
    



if __name__ == "__main__":
    main()