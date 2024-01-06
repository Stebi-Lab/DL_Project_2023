
# Deep Learning Project 2023-2024

# Effect of ESG Company data on the long term Market prospects of Companies.

As a framework to evaluate companies’ sustain-
ability performance, ESG(Environmental, Social
and Governance) is getting increasing attentions
worldwide. In this study, we have investigated
the application of both deep learning and non-
deep-learning method to find the correspondence
between ESG data and stock price of listed com-
panies taking into account the long-term influ-
ence of ESG data. We firstly identified the exist-
ing problem which could be solved with the help
of deep learning technics based on literature re-
view and preliminary analysis. Then relevant data
from public source including ESG performance
data and stock price data of a group of enterprise
were collected to build the dataset fit for further
analysis. An architecture to simplify the learning
of sequential data was proposed and two types
of models, one non-deep-learning method and
one deep learning method, were established. Fi-
nally, we compared the results of the two methods
and found that 1) the LSTM method could effec-
tively higher capacity to accommodate ESG data
2) LSTM method considering ESG data provided
better prediction resul

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

