import torch
import numpy as np
a = torch.randn([3, 3, 3,3])
#a = a.numpy()
#a = np.arange(27)
#a = a.reshape(3,3,3)
b = a.clone()
c = a.clone()
print(a)

idx = torch.Tensor([[0,0], [0,1], [0, 2]]).to(torch.long)

#print(idx[:,0])
#print(idx[:,1])
a0 = a[..., 0, 0]
a1 = a[..., 0, 1]
a2 = a[...,0, 2]
a3 = a[...,1, 0]
a4 = a[...,1, 1]
a5 = a[...,1, 2]
a6 = a[...,2, 0]
a7 = a[...,2, 1]
a8 = a[...,2, 2]
print(a0)
print(a1)
b[..., 0, 0] = a3
b[..., 1, 0] = a6
b[..., 2, 0] = a7
b[..., 0, 1] = a0
b[..., 1, 1] = a4
b[..., 2, 1] = a8
b[..., 0, 2] = a1
b[..., 1, 2] = a2
b[..., 2, 2] = a5

c[..., 0, 0] = a1
c[..., 1, 0] = a0
c[..., 2, 0] = a3
c[..., 0, 1] = a2
c[..., 1, 1] = a4
c[..., 2, 1] = a6
c[..., 0, 2] = a5
c[..., 1, 2] = a8
c[..., 2, 2] = a7
print(a)
print(b)
print(c)
