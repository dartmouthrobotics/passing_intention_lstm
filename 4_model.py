import torch.nn as nn
import yaml

CONFIG_PATH = "./param/lstm_config.yaml"

with open(CONFIG_PATH, 'r') as file:
    config = yaml.safe_load(file)
num_features_ = config['num_features']
num_classes_ = config['num_classes']
hidden_size_ = config['hidden_size']
num_layers_ = config['num_layers']
dropout_fraction_ = config['dropout_fraction']

class TimeSeriesClassifier(nn.Module):
    def __init__(
        self,
        num_features,
        num_classes,
        hidden_size=hidden_size_,
        num_layers=num_layers_,
        dropout_fraction=dropout_fraction_,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_fraction,
        )

        self.classifier = nn.Linear(in_features=hidden_size, 
                                    out_features=num_classes)
        # self.sigmoid = nn.Sigmoid() # if we use one integer output
        # https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html
        # self.softmax = nn.Softmax(dim=1) # make probability of L, R distribution result as sum 1

    def forward(self, x):
        # self.lstm.flatten_parameters() # for multi-gpu training purpose. not important
        # init hidden can be skipped
        # https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
        _output, (hidden_n, _cell_state_n) = self.lstm(x) 
        # print("hidden", hidden_n[-1], hidden_n[-1].shape)
        x = self.classifier(hidden_n[-1])
        # x = self.sigmoid(x)
        # x = self.softmax(x) # CrossEntropyLoss no need
        return x # length 2 array for each x


if __name__ == "__main__":
    ### USAGE
    import torch

    X = torch.randn([1, 10, 7])
    print("input", X)
    model = TimeSeriesClassifier(num_features=num_features_, num_classes=num_classes_)

    with torch.no_grad():
        pred = model(X)
        print("output", pred)
