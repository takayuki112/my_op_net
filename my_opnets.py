import torch
import torch.nn as nn
import torch.optim as optim

class StackedBranchNet(nn.Module):
    def __init__(self, num_sensors, num_hidden, hidden_size, output_size):
        '''
        num_sensors - we decide in m fixed values of x to sample from different u(x)s
        Here all these inputs share a common set of parameters
        
        '''
        super("StackedBranchNet").__init__()
        
        self.input_size = num_sensors
        self.num_hidden = num_hidden
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # NN Architecture ~~
        
        self.input_fc = nn.Linear(self.input_size, hidden_size)
        
        self.hidden_layers = nn.ModuleList()
        for i in range(num_hidden):
            hfc = nn.Linear(hidden_size, hidden_size)
            self.hidden_layers.append(hfc)
        
        self.output_fc = nn.Linear(hidden_size, output_size)
        
        # Activation is just LeakyReLU for now
        self.activation = nn.LeakyReLU
        
        
    def forward(self, x):
        out = self.input_fc(x)
        for layer in self.hidden_layers:
            out = layer(out)
            out = self.activation(out)
        out = self.output_fc(out)
        return out
    
    
class TrunkNet(nn.Module):
    def __init__(self, num_hidden, hidden_size, output_size):
        super("TrunkNet").__init__()
        
        self.num_hidden = num_hidden
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # NN Architecture ~~
        
        self.input_fc = nn.Linear(1, hidden_size)
        
        self.hidden_layers = nn.ModuleList()
        for i in range(num_hidden):
            hfc = nn.Linear(hidden_size, hidden_size)
            self.hidden_layers.append(hfc)
        
        self.output_fc = nn.Linear(hidden_size, output_size)
        
        # Activation is just LeakyReLU for now
        self.activation = nn.LeakyReLU
        
        
    def forward(self, x):
        out = self.input_fc(x)
        for layer in self.hidden_layers:
            out = layer(out)
            out = self.activation(out)
        out = self.output_fc(out)
        return out     
    
   
class YoOpNet(nn.Module):
    def __init__(self):
        super("YoOpNet").__init__()
        
        self.num_sensors = 10
        self.num_hidden_branch = 1
        self.hidden_size_branch = 50
        self.output_size_branch = 50
        
        self.num_hidden_trunk = 1
        self.hidden_size_trunk = 30
        self.output_size_trunk = 1
        
        
        self.branch_net = StackedBranchNet(self.num_sensors, self.num_hidden_branch, self.hidden_size_branch, self.output_size_branch)
        self.trunk_net = TrunkNet(self.num_hidden_trunk, self.hidden_size_trunk, self.output_size_trunk)
        
        self.combined_fc = nn.Linear(self.output_size_branch + self.output_size_trunk, 1)
    
    def forward(self, x_brnach, x_trunk):
        out_branch = self.branch_net(x_brnach)
        out_trunk = self.trunk_net(x_trunk)
        combined = torch.cat(out_branch, out_trunk)
        out = self.combined_fc(combined)
        return out