import torch.nn as nn


class TimeSeriesClassifier(nn.Module):
    def __init__(self, num_features, num_classes, hidden_size=256, num_layers=3):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.4,
        )

        self.classifier = nn.Linear(in_features=hidden_size, out_features=num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        self.lstm.flatten_parameters()
        _output, (hidden_n, _cell_state_n) = self.lstm(x)
        x = self.classifier(hidden_n[-1])
        x = self.sigmoid(x)
        return x


if __name__ == "__main__":
    ### USAGE
    import torch

    X = torch.randn([1, 3000, 6])
    model = TimeSeriesClassifier(num_features=6, num_classes=2)

    with torch.no_grad():
        pred = model(X)
        print(pred)
