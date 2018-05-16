import unittest

import torch
import numpy as np
import numpy.testing as npt

from chrono_initialization import init as chrono_init


def get_biases(model):
    for name, p in model.named_parameters():
        if 'bias' in name:
            return p.data
    

class TestChronoInitialization(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.optimizer = torch.optim.Adam
        cls.lr = 0.001
        cls.t = torch.cat([torch.eye(10) for i in range(10)])
        cls.iter_num = 10000
    
    def setUp(self):
        pass

    def test_check_forget_gate_of_LSTM(self):
        model = torch.nn.LSTM(10, 10, 2)
        before_biases = get_biases(model)        
        model = chrono_init(model, 1, 10)
        after_biases = get_biases(model)
        assert before_biases[0] != after_biases[0]
        
    def test_check_forget_gate_of_GRU(self):    
        model = torch.nn.GRU(10, 8, 2)
        before_biases = get_biases(model)
        model = chrono_init(model, 2, 5)
        after_biases = get_biases(model)
        assert before_biases[0] != after_biases[0]

    @unittest.skip('WIP')
    def test_functinal_test_of_LSTM(self):
        # sin wave to sin wave
        model = torch.nn.Sequential(
            torch.nn.LSTM(1, 10, 1, batch_first=True),
            )
        B = 10
        x = np.linspace(0, 2*np.pi, 60)
        x = torch.stack([torch.Tensor(x) for i in range(B)])        
        t = torch.stack([torch.Tensor(x) for i in range(B)])
        
        x = x.view(B, -1, 1)
        t = x.view(B, -1, 1)
        
        optim = self.optimizer(model.parameters(), self.lr)
        model.train()
        
        for i in range(self.iter_num):
            optim.zero_grad()
            o = model(x)
            loss = torch.nn.functional.mse_loss(o, t)
            loss.backward()
            optim.step()
        model.eval()
        x = torch.stack([torch.eye(10) for i in range(1)])
        o, h = model(x)        
        t = torch.stack([torch.eye(10) for i in range(1)])        
        npt.assert_almost_equal(t.numpy(), o.detach().numpy())
        
        

        
    
if __name__ == '__main__':
    unittest.run()
