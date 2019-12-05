import numpy as np

class Dual:
    '''
    Forward mode autodiff for scalars
    '''
    
    def __init__(self, value=0, grad=0):
        self.value = value
        self.grad = grad   
    
    def __add__(self, dual):
        return Dual(self.value+dual.value, self.grad+dual.grad)
    
    def __sub__(self, dual):
        return Dual(self.value-dual.value, self.grad-dual.grad)

    def __mul__(self, dual):
        return Dual(self.value*dual.value, self.value*dual.grad+self.grad*dual.value)

    def __pow__(self, dual):
        return Dual(self.value**dual.value, dual.value*self.value**(dual.value-1)*self.grad)

    def __truediv__(self, dual):
        return Dual(self.value/dual.value, self.grad*dual.value-self.value*dual.grad/self.value**2)

    

