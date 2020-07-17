class AE(nn.Module):
    def __init__(self, input_features, out_features):
        super().__init__()
        self.encoder_hidden_layer = nn.Linear(
            in_features=input_features, out_features=out_features
        )
        self.encoder_output_layer = nn.Linear(
            in_features=128, out_features=out_features)
        )
        self.decoder_hidden_layer = nn.Linear(
            in_features=128, out_features=out_features)
        )
        self.decoder_output_layer = nn.Linear(
            in_features=out_features), out_features=in_features
        )

    def forward(self, features):
        activation = self.encoder_hidden_layer(features)
        activation = torch.relu(activation)
        code = self.encoder_output_layer(activation)
        code = torch.relu(code)
        activation = self.decoder_hidden_layer(code)
        activation = torch.relu(activation)
        activation = self.decoder_output_layer(activation)
        reconstructed = torch.relu(activation)
        return reconstructed
