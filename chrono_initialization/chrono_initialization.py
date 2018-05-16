import numbers

import torch


def init(rnn: torch.nn.Module, Tmax=None, Tmin=1):
    '''chrono initialization(Ref: https://arxiv.org/abs/1804.11188)
    '''
    
    assert isinstance(Tmin, numbers.Number), 'Tmin must be numeric.'
    assert isinstance(Tmax, numbers.Number), 'Tmax must be numeric.'    
    for name, p in rnn.named_parameters():
        if 'bias' in name:
            n = p.nelement()
            hidden_size = n // 4            
            p.data.fill_(0)
            if isinstance(rnn, (torch.nn.LSTM, torch.nn.LSTMCell)):
                p.data[hidden_size: 2*hidden_size] = \
                    torch.log(torch.nn.init.uniform_(p.data[0: hidden_size], 1, Tmax - 1))
                # forget gate biases = log(uniform(1, Tmax-1))
                p.data[0: hidden_size] = -p.data[hidden_size: 2*hidden_size]
                # input gate biases = -(forget gate biases)

    return rnn
