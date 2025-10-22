import torch
from torch import nn


class AE(nn.Module):
    def __init__(self, input_shape: int, out_features: int):
        """
        A basic and simple to use AutoEncoder with two layers in both the encoder and decoder.

        :param input_shape: The number of features for input and the final reconstruction.
        :type input_shape: int
        :param out_features: The number of output features in the bottleneck (latent space representation).
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

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Runs the full forward pass on the model from input features to reconstructed output.
        In practice, this will only be done during training.

        :param features: The input tensor of data features, typically of shape (batch_size, input_shape).
        :type features: torch.Tensor
        :return: The reconstructed output tensor, with the same shape as the input (batch_size, input_shape).
        :rtype: torch.Tensor

        Example:
            .. code-block:: python

                auto_model = AE(10, 4)
                x = torch.rand(2, 10)  # batch_size, n_features
                result = auto_model(x)
                print(result.shape)  # (2, 10)
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

    def generate_representation(self, features: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass only through the encoder part of the AutoEncoder to generate the latent representation (code).

        :param features: The input tensor of data features.
        :type features: torch.Tensor
        :return: The latent space representation (code) tensor, of shape (batch_size, out_features).
        :rtype: torch.Tensor
        """
        activation = self.encoder_hidden_layer(features)
        activation = torch.relu(activation)
        code = self.encoder_output_layer(activation)
        code = torch.relu(code)
        return code