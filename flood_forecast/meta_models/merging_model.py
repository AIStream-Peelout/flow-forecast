class MergingModel(torch.nn.Module):
  def __init__(self, method, other_params):
    super().__init__()
    self.method_dict = {"Bilinear":torch.nn.Bilinear}
    self.method_layer = self.method_dict[method](**other_params)

  def forward(self, x):
    x = self.method_layer(x)
    return x
