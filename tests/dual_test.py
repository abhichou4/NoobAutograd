import sys
sys.path.append('/home/abhineet/workspace')

from NoobAutograd.dual import Dual

def power_test():

    d1 = Dual(2,1)
    d2 = Dual(5)
    d3 = Dual(3)
    d4 = d1**d3
    assert d4.value==8 and d4.grad==12

def main():
    power_test()

if __name__ == '__main__':
    main()
