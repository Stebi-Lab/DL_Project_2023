
# Deep Learning Project 2023-2024

# Effect of ESG Company data on the long term Market prospects of Companies.

As a framework to evaluate sustainability performance 
of companies, ESG(Environmental, Social
and Governance) is getting increasing attention
worldwide. In this study, we have investigated
the application of both deep learning and non
deep-learning methods to find the correspondence
between ESG data and stock price of listed companies 
considering the long-term influence of ESG
data. Relevant data from public source including
ESG performance data and stock price data of a
group of companies were collected to build the
data set. An architecture to simplify the learning 
of sequential data was proposed. Normal regression 
and LSTM models were trained to predict stock price 
with/without considering ESG
features. The result of the study indicate that i)
LSTM models incorporate ESG data better than
non-DL models and ii) when using LSTM models,
the training with ESG data was able to provide
better prediction than that without ESG data

# Reproduction of Results:

# Training:

To start training the variable **useModel** must be **false**.
It is recommended to also set **plotting** to **false** as this will stop ploting the inference for each company.
If wanted by setting **safe** to **True** the model will be safed on the root folder.
More than 15 companies can be tested if the iterator break is removed. 
The variable **ESG** will define if ESG data will be embedded into the training data or not.
It is important to note that pytorchs LSTM training is non deterministic. 

```python
def main():

    if 0:
        print("Hello")

    # Parameters for safing model 
    safe = False
    safeModelName = "Model"

    # Parameters to use the Model for inference and prediction on the data.
    useModel = True
    Model = "Model_NoESG_1" #Model to use

    doTraining = False
    plotting = True

    if useModel:
        lstm.load_state_dict(torch.load(Model))
        doTraining = False
        plotting = True
```

# Plotting and Evaluation:

To use an already exisiting model set the **useModel** to **True** and set **plotting** to **True**.
This will iterate over the companies creating the plots from the report. 

The results from the paper are in the results.txt file together with the names of the trained models. All models are in the root folder.
