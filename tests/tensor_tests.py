import sys
sys.path.append('/home/abhineet/workspace/')

from NoobAutograd.tensor import *

def tensor_sum_test():

    t = Tensor([1,2,3,4], requires_grad=True)
    s = t.sum()
    s.backward(Tensor(1.))
    assert s.data == 10
    assert s.grad.data==1 and t.grad.data.all()

def tensor_add_test():

    t1 = Tensor([1,2,3], requires_grad=True)
    t2 = Tensor([1,2,3], requires_grad=True)
    t3 = Tensor(1, requires_grad=True)

    s1 = t1 + t2
    s1.backward(Tensor(np.ones(s1.shape)))  
    assert t1.grad.data.all() and t2.grad.data.all()
    s2 = t1 + t3
    s2.backward(Tensor(np.ones(s2.shape)))
    assert t1.grad.data.all() and (t2.grad.data/2).all() and t3.grad.data==3

def main():
    tensor_sum_test()
    tensor_add_test()

if __name__ == '__main__':
    main()