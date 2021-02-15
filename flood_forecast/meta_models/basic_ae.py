import torch
from torch import nn


class AE(nn.Module):
    def __init__(self, input_shape: int, out_features: int):
        """A basic and simple to use AutoEncoder

        :param input_shape: The number of features for input.
        :type input_shape: int
        :param out_features: The number of output features (that will be merged)
        :type out_features: int
        """
        super().__init__()
        self.encoder_hidden_layer = nn.Linear(
            in_features=input_shape, out_features=out_features
        )
        self.encoder_output_layer = nn.Linear(
            in_features=out_features, out_features=out_features
        )
        self.decoder_hidden_layer = nn.Linear(
            in_features=out_features, out_features=out_features
        )
        self.decoder_output_layer = nn.Linear(
            in_features=out_features, out_features=input_shape
        )

    def forward(self, features: torch.Tensor):
        """Runs the full forward pass on the model. In practice
        this will only be done during training.

        :param features: [description]
        :type features: [type]
        :return: [description]
        :rtype: [type]

        .. code-block:: python
            auto_model = AE(10, 4)
            x = torch.rand(2, 10) # batch_size, n_features
            result = auto_model(x)
            print(result.shape) # (2, 10)
        """
        activation = self.encoder_hidden_layer(features)
        activation = torch.relu(activation)
        code = self.encoder_output_layer(activation)
        code = torch.relu(code)
        activation = self.decoder_hidden_layer(code)
        activation = torch.relu(activation)
        activation = self.decoder_output_layer(activation)
        reconstructed = torch.relu(activation)
        return reconstructed

    def generate_representation(self, features):
        activation = self.encoder_hidden_layer(features)
        activation = torch.relu(activation)
        code = self.encoder_output_layer(activation)
        code = torch.relu(code)
        return code
