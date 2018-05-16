# Chrono Initialization (Pytorch implementation)


### Dependency

* pytorch > 0.4.0

### How to use 


```python
from chrono_initialization import init as chrono_init

model = LSTM(...)
model = chrono_init(model)

model = GRU(...)
model = chrono_init(model)

```


### Reference

* https://openreview.net/forum?id=SJcKhk-Ab