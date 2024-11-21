
import  torch
import torch.nn as nn
from torch.nn import functional as F
from torch import fx
from op import  *
from  from_torch import *
# import from_torch
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.weight = nn.Parameter(torch.randn(128, 128))
    def forward(self, x):
        x = torch.matmul(x, self.weight)
        x = torch.relu(x)
        return x
    
module =  MyModel()
module = module.eval()
fx_module = fx.symbolic_trace(module)
IR_module = from_torch(
    fx_module,
    input_shapes = [(1, 128)],
    call_function_map = {
      torch.matmul: map_matmul,
      torch.relu: map_relu,
    },
    call_module_map={},
)
IR_module.show()
print(type(IR_module))
