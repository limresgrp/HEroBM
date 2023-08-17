import torch

class Data(dict):
    def __init__(self, *args, **kw):
        super(Data,self).__init__(*args, **kw)
        
    def __setitem__(self, key, value):
        super(Data,self).__setitem__(key, value)
    
    def values(self):
        return [self[key] for key in self]  
    
    def itervalues(self):
        return (self[key] for key in self)
    
    def to(self, device):
        for k, v in self.items():
            assert isinstance(v, torch.Tensor)
            self.__setitem__(k, v.to(device))
        return self