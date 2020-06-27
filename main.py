import torch
from torch.autograd import Variable
from groupy.gconv.pytorch_gconv import P4ConvZ2, P4ConvP4, P4MConvZ2, P4MConvP4M, P4MConvP4M_SC,P4MConvP4M_SCC, P4MConvP4M_SF

# Construct G-Conv layers
#C1 = P4ConvZ2(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
#C2 = P4ConvP4(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)

P1 = P4MConvZ2(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
P2 = P4MConvP4M(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
P3 = P4MConvP4M_SF(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)

C1 = P4MConvZ2(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
C2 = P4MConvP4M(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1)
C3 = P4MConvP4M_SC(in_channels=64, out_channels=22, kernel_size=3, stride=1, padding=1)
C4 = P4MConvP4M_SC(in_channels=22, out_channels=45, kernel_size=3, stride=1, padding=1)
C5 = P4MConvP4M_SCC(in_channels=45, out_channels=65, kernel_size=3, stride=1, padding=1)
C6 = P4MConvP4M_SF(in_channels=65, out_channels=65, kernel_size=3, stride=1, padding=1)

# Create 10 images with 3 channels and 9x9 pixels:
x = Variable(torch.randn(1, 3, 32, 32))
#x = Variable(torch.zeros((10, 3, 9, 9)).int().random_(0,255))
# fprop
#y = C1(x)
#print(y.data.shape)  # (10, 64, 4, 9, 9)
#y = C2(y)
#print(y.data.shape)  # (10, 64, 4, 9, 9)
#y = C3(y)
#y = C4(y)
#y = C5(y)
#print(y.data.shape)  # (10, 64, 4, 9, 9)

z = P1(x)
print(z.data.shape)  # (10, 64, 8, 9, 9)
z = P2(z)
#print(z.data.shape)  # (10, 64, 8, 9, 9)
print('HERE!!!!!!!!!!!!!!!!!!!')  # (10, 64, 8, 9, 9)
z = P3(z)
print(z.data.shape)  # (10, 64, 8, 9, 9)
#y = C2(C1(x))
#print(y.data.shape)  # (10, 64, 4, 9, 9)
#print(x[0][0])
#print(z[0][0])