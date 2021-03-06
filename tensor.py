import numpy as np

class Dependencies:
    '''
    Class that stores
    1. Tensors that the current tensor depends on
    2. Operations those came form 
    '''
    def __init__(self, tensor, grad_fn):
        self.tensor = tensor
        self.grad_fn = grad_fn  

class Tensor:
    '''
    Wraps numpy array
    '''

    def __init__(self, data, requires_grad=False, depends_on=[]):
        '''
        data :          numpy arrays 
        requires_grad:  bool (is it part of the backward graph or not)
        depends_on:     List of Dependencies 
        '''
        self.requires_grad = requires_grad
        self.depends_on = depends_on
        self.grad = None

        if isinstance(data, np.ndarray):
            self.data = data
        else:
            self.data = np.array(data)

        self.shape = self.data.shape

        if self.requires_grad:
            self.zero_grad()

    def __repr__(self):

        return "data: {}\nrequires_grad: {}\n".format(self.data, self.requires_grad)

    def zero_grad(self):
        '''
        Gradients are initially zeros and 
        can be cleaned when needed to be re-computed 
        '''
        self.grad = Tensor(np.zeros(self.shape))


    def backward(self, grad):
        '''
        This tensor is capable of propogating 
        the gradien backwards
        '''
        assert self.requires_grad, "tensor not part of backwards graph"
        self.grad.data += grad.data #seed value

        for dependency in self.depends_on: #recursivly all gradients computed
            grad_val = dependency.grad_fn(grad.data)
            dependency.tensor.backward(Tensor(grad_val))

    def sum(self):
        '''
        Take an N-d Array and returns sum its of elements 
        '''
        data = self.data.sum()
        requires_grad = self.requires_grad

        if requires_grad:
            def grad_fn(grad):
                return grad * np.ones((self.shape)) 
            
            depends_on = [Dependencies(self, grad_fn)]

        else:
            depends_on = []

        return Tensor(data, requires_grad, depends_on) 

    def add(self, t):
        '''
        Returns sum of two tensors
        '''
        data = np.add(self.data, t.data)
        requires_grad = self.requires_grad or t.requires_grad

        depends_on = [] 
        
        if self.requires_grad:
            def grad_fn(grad):
                # Handle broadcasting
                ndims_added = grad.data.ndim - self.data.ndim
                for _ in range(ndims_added):
                    grad.sum(axis=0)
                return grad 
            depends_on.append(Dependencies(self, grad_fn))
        
        if t.requires_grad:
            def grad_fn(grad):
                # Handle broadcasting
                ndims_added = grad.data.ndim - t.data.ndim 
                for _ in range(ndims_added):
                    grad = grad.sum()
                return grad
            depends_on.append(Dependencies(t, grad_fn)) 

        return Tensor(data, requires_grad, depends_on)

    def __add__(self, t):
        return self.add(t)
        
