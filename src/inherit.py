#!/usr/bin/env python
import utils

class A:
    def __init__(self, shared_var):
        "docstring"
        self.shared_var = shared_var
        self.varA = 1
        
class B(A):
    def __init__(self, shared_var):
        "docstring"
        super().__init__(shared_var)
        self.varA = 2
        self.varB = 3

class C(A):
    def __init__(self, shared_obj):
        "docstring"
        super().__init__(shared_obj.shared_var)
        # self.shared_var = 5
        print(self.shared_var)
        

def main():
    a = A(2)
    b = B(a)
    c = C(a)
    print(c.shared_var)

if __name__ == '__main__':
    main()
