import torch
import torch.nn as nn
import torch.nn.parallel
import numpy as np
from torch.nn.parameter import Parameter
import time

class rational_shuffle_true(nn.Module):
    def __init__(self):
        super(rational_shuffle_true, self).__init__()
        self.a0 = Parameter(torch.FloatTensor([0.0218]))
        self.a1 = Parameter(torch.FloatTensor([0.5000]))
        self.a2 = Parameter(torch.FloatTensor([1.5957]))
        self.a3 = Parameter(torch.FloatTensor([1.1915]))
        self.b0 = Parameter(torch.FloatTensor([1.0000]))
        self.b1 = Parameter(torch.FloatTensor([0.0000]))
        self.b2 = Parameter(torch.FloatTensor([2.3830]))
    def forward(self, x):
        y = (self.a3*x**3+self.a2*x**2+self.a1*x+self.a0)/(self.b2*x**2+self.b1*x+self.b0)
        return y

class mlp_shuffle_true(nn.Module):
    def __init__(self, n_hidden, input_size=4, output_size=4):
        super(mlp_shuffle_true, self).__init__()
        layers = [nn.Linear(input_size, n_hidden),
                  rational_shuffle_true(),
                  nn.Linear(n_hidden, output_size)]
        self.nn = nn.Sequential(*layers)
        # self.flag = 0
        # self.avg = 0.0
    def forward(self, p):
        # ## 
        # a = self.nn[0]
        # b = a(p).detach().cpu().numpy()
        # if self.flag == 0:
        #     self.b = b
        # else:
        #     self.b = np.concatenate((self.b, b), axis=0)
        # self.flag += 1
        # # print(b.shape) # (150, 1024)
        # # print(self.flag)
        # if(self.flag == 500):
        #     np.save("./before_activate.npy", self.b)
        #     self.flag = 0
        # # print("before activate !!!!!!!!")
        
        
        # self.flag += 1
        # if self.flag == 500:
        #     a = self.nn[0]
        #     b = self.nn[1]
        #     u, s, v = torch.svd(b(a(p)))
        #     s = s.detach().cpu().numpy()
        #     s = np.sqrt(s)
        #     # print('shape')
        #     # print(s.shape)
        #     s_sum = np.sum(s)
        #     # print('sum')
        #     # print(s_sum)
        #     s_max = np.max(s, 0)
        #     # print('max')
        #     # print(s_max)
        #     # print('func S')
        #     print(s_sum/s_max)
        #     self.flag = 0
        #     self.avg += s_sum/s_max
        #     # time.sleep(3)

        y = self.nn(p)
        return y

class NeurVec_shuffle_true(nn.Module):
    def __init__(self):
        super(NeurVec_shuffle_true, self).__init__()
        self.error = mlp_shuffle_true(n_hidden=1024, input_size=4, output_size=4)
        # self.error = mlp_shuffle_true(n_hidden=364, input_size=4, output_size=4)








        # noise = np.random.normal(0, 0.001, self.data.shape).astype(dtype=np.float32)
        # self.data += noise
class rational_shuffle_true_nihe(nn.Module):
    def __init__(self):
        super(rational_shuffle_true_nihe, self).__init__()
        self.a0 = Parameter(torch.FloatTensor([0.0140]))
        self.a1 = Parameter(torch.FloatTensor([-0.1000]))
        self.a2 = Parameter(torch.FloatTensor([2.5]))
        self.a3 = Parameter(torch.FloatTensor([0.0070]))


    def forward(self, x):
        sig = torch.sigmoid(self.a2*x)
        y = self.a0*x + self.a1*sig*(1-sig) + self.a3
        return y

class mlp_shuffle_true_nihe(nn.Module):
    def __init__(self, n_hidden, input_size=4, output_size=4):
        super(mlp_shuffle_true_nihe, self).__init__()
        layers = [nn.Linear(input_size, n_hidden),
                  rational_shuffle_true_nihe(),
                  nn.Linear(n_hidden, output_size)]
        self.nn = nn.Sequential(*layers)
        self.flag = 0
    def forward(self, p):
        y = self.nn(p)
        return y

class NeurVec_shuffle_true_nihe(nn.Module):
    def __init__(self):
        super(NeurVec_shuffle_true_nihe, self).__init__()
        self.error = mlp_shuffle_true_nihe(n_hidden=1024, input_size=4, output_size=4)
        # self.error = mlp_shuffle_true(n_hidden=364, input_size=4, output_size=4)


class mlp_shuffle_true_mlp(nn.Module):
    def __init__(self, n_hidden, input_size=4, output_size=4):
        super(mlp_shuffle_true_mlp, self).__init__()
        layers = [nn.Linear(input_size, n_hidden),
                nn.Linear(n_hidden, output_size)]
        self.nn = nn.Sequential(*layers)

    def forward(self, p):
        y = self.nn(p)
        return y

class NeurVec_shuffle_true_mlp(nn.Module):
    def __init__(self):
        super(NeurVec_shuffle_true_mlp, self).__init__()
        self.error = mlp_shuffle_true_mlp(n_hidden=1024, input_size=4, output_size=4)






class rational_shuffle_true_double(nn.Module):
    def __init__(self):
        super(rational_shuffle_true_double, self).__init__()
        self.a0 = Parameter(torch.FloatTensor([0.0140]))
        self.a1 = Parameter(torch.FloatTensor([-0.1000]))
        self.a2 = Parameter(torch.FloatTensor([2.5]))
        self.a3 = Parameter(torch.FloatTensor([0.0070]))
    def forward(self, x):
        sig = torch.sigmoid(self.a2*x)
        y = self.a0*x + self.a1*sig*(1-sig) + self.a3
        return y
class rational_shuffle_true_double_2(nn.Module):
    def __init__(self):
        super(rational_shuffle_true_double_2, self).__init__()
        self.a0 = Parameter(torch.FloatTensor([0.0140]))
        self.a1 = Parameter(torch.FloatTensor([-0.1000]))
        self.a2 = Parameter(torch.FloatTensor([2.5]))
        self.a3 = Parameter(torch.FloatTensor([0.0070]))
    def forward(self, x):
        sig = torch.sigmoid(self.a2*x)
        y = self.a0*x + self.a1*sig*(1-sig) + self.a3
        return y
class mlp_shuffle_true_double(nn.Module):
    def __init__(self, n_hidden, input_size=4, output_size=4):
        super(mlp_shuffle_true_double, self).__init__()
        layers = [nn.Linear(input_size, n_hidden),
                  rational_shuffle_true_double(),
                  nn.Linear(n_hidden, output_size)]

        layers2= [nn.Linear(output_size, n_hidden//4),
                  rational_shuffle_true_double_2(),
                  nn.Linear(n_hidden//4, output_size)]
        self.nn = nn.Sequential(*layers)
        self.nn2= nn.Sequential(*layers2)
    def forward(self, p):
        y = self.nn(p)
        y = y + self.nn2(y)
        return y

class NeurVec_shuffle_true_double(nn.Module):
    def __init__(self):
        super(NeurVec_shuffle_true_double, self).__init__()
        self.error = mlp_shuffle_true_double(n_hidden=1024, input_size=4, output_size=4)




class rational_shuffle_true_channelwise(nn.Module):
    def __init__(self):
        super(rational_shuffle_true_channelwise, self).__init__()
        # self.a0 = Parameter(torch.FloatTensor([0.0140]))
        self.a0 = Parameter(torch.FloatTensor(torch.full((1,1024),0.0140)))
        self.a1 = Parameter(torch.FloatTensor(torch.full((1,1024),-0.1000)))
        self.a2 = Parameter(torch.FloatTensor(torch.full((1,1024),2.5)))
        self.a3 = Parameter(torch.FloatTensor(torch.full((1,1024),0.0070)))

    def forward(self, x):
        # (batch size, channels)
        sig = torch.sigmoid(self.a2*x)
        y = self.a0*x + self.a1*sig*(1-sig) + self.a3
        return y

class mlp_shuffle_true_channelwise(nn.Module):
    def __init__(self, n_hidden, input_size=4, output_size=4):
        super(mlp_shuffle_true_channelwise, self).__init__()
        layers = [nn.Linear(input_size, n_hidden),
                  rational_shuffle_true_channelwise(),
                  nn.Linear(n_hidden, output_size)]
        self.nn = nn.Sequential(*layers)
    def forward(self, p):
        y = self.nn(p)
        return y

class NeurVec_shuffle_true_channelwise(nn.Module):
    def __init__(self):
        super(NeurVec_shuffle_true_channelwise, self).__init__()
        self.error = mlp_shuffle_true_channelwise(n_hidden=1024, input_size=4, output_size=4)





class rational_shuffle_true_4channelwise(nn.Module):
    def __init__(self):
        super(rational_shuffle_true_4channelwise, self).__init__()
        self.a0 = Parameter(torch.FloatTensor(torch.full((1,4),0.0140)))
        self.a1 = Parameter(torch.FloatTensor(torch.full((1,4),-0.1000)))
        self.a2 = Parameter(torch.FloatTensor(torch.full((1,4),2.5)))
        self.a3 = Parameter(torch.FloatTensor(torch.full((1,4),0.0070)))
    def forward(self, x):
        sig = torch.sigmoid(self.a2*x)
        y = self.a0*x + self.a1*sig*(1-sig) + self.a3
        return y

class mlp_shuffle_true_4channelwise(nn.Module):
    def __init__(self, n_hidden, input_size=4, output_size=4):
        super(mlp_shuffle_true_4channelwise, self).__init__()
        layers = [nn.Linear(input_size, n_hidden),
                  rational_shuffle_true_4channelwise(),
                  nn.Linear(n_hidden, output_size)]
        self.nn = nn.Sequential(*layers)
    def forward(self, p):
        y = self.nn(p)
        return y

class NeurVec_shuffle_true_4channelwise(nn.Module):
    def __init__(self):
        super(NeurVec_shuffle_true_4channelwise, self).__init__()
        self.error = mlp_shuffle_true_4channelwise(n_hidden=4, input_size=4, output_size=4)






class rational_shuffle_true_channelwiseFine(nn.Module):
    def __init__(self):
        super(rational_shuffle_true_channelwiseFine, self).__init__()
        self.a0 = Parameter(torch.FloatTensor(torch.full((1,1024),0.0140)))
        self.a1 = Parameter(torch.FloatTensor(torch.full((1,1024),-0.1000)))
        self.a2 = Parameter(torch.FloatTensor(torch.full((1,1024),2.5)))
        self.a3 = Parameter(torch.FloatTensor(torch.full((1,1024),0.0070)))

    def forward(self, x):
        # print(x.shape)        # (16384, 1024) (batch size, channels)
        sig = torch.sigmoid(self.a2*x)
        y = self.a0*x + self.a1*sig*(1-sig) + self.a3
        return y

class mlp_shuffle_true_channelwiseFine(nn.Module):
    def __init__(self, n_hidden, input_size=4, output_size=4):
        super(mlp_shuffle_true_channelwiseFine, self).__init__()
        layers = [nn.Linear(input_size, n_hidden),
                  rational_shuffle_true_channelwiseFine(),
                  nn.Linear(n_hidden, output_size)]
        self.nn = nn.Sequential(*layers)
    def forward(self, p):
        y = self.nn(p)
        return y

class NeurVec_shuffle_true_channelwiseFine(nn.Module):
    def __init__(self):
        super(NeurVec_shuffle_true_channelwiseFine, self).__init__()
        self.error = mlp_shuffle_true_channelwiseFine(n_hidden=1024, input_size=12, output_size=12)




class rational_shuffle_true_groupchannelwise(nn.Module):
    def __init__(self):
        super(rational_shuffle_true_groupchannelwise, self).__init__()
        self.a0 = Parameter(torch.FloatTensor(torch.full((1,64),0.0140)))
        self.a1 = Parameter(torch.FloatTensor(torch.full((1,64),-0.1000)))
        self.a2 = Parameter(torch.FloatTensor(torch.full((1,64),2.5)))
        self.a3 = Parameter(torch.FloatTensor(torch.full((1,64),0.0070)))

    def forward(self, x):
        # print(x.shape)        # (16384, 1024) (batch size, channels)
        aa0 = self.a0.repeat(1,16)
        aa1 = self.a1.repeat(1,16)
        aa2 = self.a2.repeat(1,16)
        aa3 = self.a3.repeat(1,16)
        sig = torch.sigmoid(aa2*x)
        y = aa0*x + aa1*sig*(1-sig) + aa3
        return y
        # print(aa0[0][0]) # 以128为一个单位
        # print(aa0[0][1])
        # print(aa0[0][2])
        # print(aa0[0][128])
        # print(aa0[0][256])  

class mlp_shuffle_true_groupchannelwise(nn.Module):
    def __init__(self, n_hidden, input_size=4, output_size=4):
        super(mlp_shuffle_true_groupchannelwise, self).__init__()
        layers = [nn.Linear(input_size, n_hidden),
                  rational_shuffle_true_groupchannelwise(),
                  nn.Linear(n_hidden, output_size)]
        self.nn = nn.Sequential(*layers)
    def forward(self, p):
        y = self.nn(p)
        return y

class NeurVec_shuffle_true_groupchannelwise(nn.Module):
    def __init__(self):
        super(NeurVec_shuffle_true_groupchannelwise, self).__init__()
        self.error = mlp_shuffle_true_groupchannelwise(n_hidden=1024, input_size=4, output_size=4)



class rational_shuffle_true_channelwise_nolinear(nn.Module):
    def __init__(self):
        super(rational_shuffle_true_channelwise_nolinear, self).__init__()
        self.a1 = Parameter(torch.FloatTensor(torch.full((1,1024),-0.1000)))
        self.a2 = Parameter(torch.FloatTensor(torch.full((1,1024),2.5)))
        self.a3 = Parameter(torch.FloatTensor(torch.full((1,1024),0.0001)))
    def forward(self, x):
        # print(x.shape)        # (16384, 1024) (batch size, channels)
        sig = torch.sigmoid(self.a2*(x+self.a3))
        y = self.a1*sig*(1-sig)
        return y

class mlp_shuffle_true_channelwise_nolinear(nn.Module):
    def __init__(self, n_hidden, input_size=4, output_size=4):
        super(mlp_shuffle_true_channelwise_nolinear, self).__init__()
        layers = [nn.Linear(input_size, n_hidden),
                  rational_shuffle_true_channelwise_nolinear(),
                  nn.Linear(n_hidden, output_size)]
        self.nn = nn.Sequential(*layers)
    def forward(self, p):
        y = self.nn(p)
        return y

class NeurVec_shuffle_true_channelwise_nolinear(nn.Module):
    def __init__(self):
        super(NeurVec_shuffle_true_channelwise_nolinear, self).__init__()
        self.error = mlp_shuffle_true_channelwise_nolinear(n_hidden=1024, input_size=4, output_size=4)


class rational_shuffle_true_channelwise_pingyi(nn.Module):
    def __init__(self):
        super(rational_shuffle_true_channelwise_pingyi, self).__init__()
        # self.a0 = Parameter(torch.FloatTensor([0.0140]))
        self.a0 = Parameter(torch.FloatTensor(torch.full((1,1024),0.0140)))
        self.a1 = Parameter(torch.FloatTensor(torch.full((1,1024),-0.1000)))
        self.a2 = Parameter(torch.FloatTensor(torch.full((1,1024),2.5)))
        self.a3 = Parameter(torch.FloatTensor(torch.full((1,1024),0.0070)))
        self.a4 = Parameter(torch.FloatTensor(torch.full((1,1024),0.0100)))
    def forward(self, x):
        # print(x.shape)        # (16384, 1024) (batch size, channels)
        sig = torch.sigmoid(self.a2*(x+self.a4))
        y = self.a0*x + self.a1*sig*(1-sig) + self.a3
        return y

class mlp_shuffle_true_channelwise_pingyi(nn.Module):
    def __init__(self, n_hidden, input_size=4, output_size=4):
        super(mlp_shuffle_true_channelwise_pingyi, self).__init__()
        layers = [nn.Linear(input_size, n_hidden),
                  rational_shuffle_true_channelwise_pingyi(),
                  nn.Linear(n_hidden, output_size)]
        self.nn = nn.Sequential(*layers)
    def forward(self, p):
        y = self.nn(p)
        return y

class NeurVec_shuffle_true_channelwise_pingyi(nn.Module):
    def __init__(self):
        super(NeurVec_shuffle_true_channelwise_pingyi, self).__init__()
        self.error = mlp_shuffle_true_channelwise_pingyi(n_hidden=1024, input_size=4, output_size=4)



class rational_shuffle_true_SinChannelwise(nn.Module):
    def __init__(self):
        super(rational_shuffle_true_SinChannelwise, self).__init__()
        self.a1 = Parameter(torch.FloatTensor(torch.full((1,1024),-0.1000)))
        self.a2 = Parameter(torch.FloatTensor(torch.full((1,1024),2.5)))
        self.a3 = Parameter(torch.FloatTensor(torch.full((1,1024),0.0001)))
        self.a4 = Parameter(torch.FloatTensor(torch.full((1,1024),0.001)))
        self.a5 = Parameter(torch.FloatTensor(torch.full((1,1024),0.1000)))
    def forward(self, x):
        # print(x.shape)        # (16384, 1024) (batch size, channels)
        sig = torch.sigmoid(self.a2*(x+self.a3))
        y = self.a1*sig*(1-sig) + self.a4*torch.sin(200*self.a5*x)
        return y

class mlp_shuffle_true_SinChannelwise(nn.Module):
    def __init__(self, n_hidden, input_size=4, output_size=4):
        super(mlp_shuffle_true_SinChannelwise, self).__init__()
        layers = [nn.Linear(input_size, n_hidden),
                  rational_shuffle_true_SinChannelwise(),
                  nn.Linear(n_hidden, output_size)]
        self.nn = nn.Sequential(*layers)
        # self.flag = 0
        # self.avg = 0.0
    def forward(self, p):
        # ## 
        # b = self.nn[0](p).detach().cpu().numpy()
        # if self.flag == 0:
        #     self.b = b
        # else:
        #     self.b = np.concatenate((self.b, b), axis=0)
        # self.flag += 1
        # # print(b.shape) # (150, 1024)
        # print(self.flag)
        # if(self.flag == 500):
        #     np.save("./test/before_activate.npy", self.b)
        #     self.flag = 0
        # # print("before activate !!!!!!!!")
        
        # self.flag += 1
        # if self.flag == 500:
        #     a = self.nn[0]
        #     b = self.nn[1]
        #     u, s, v = torch.svd(b(a(p)))
        #     s = s.detach().cpu().numpy()
        #     s = np.sqrt(s)
        #     s_sum = np.sum(s)
        #     s_max = np.max(s, 0)
        #     print(s_sum/s_max)
        #     self.flag = 0
        #     self.avg += s_sum/s_max

        y = self.nn(p)
        return y

class NeurVec_shuffle_true_SinChannelwise(nn.Module):
    def __init__(self):
        super(NeurVec_shuffle_true_SinChannelwise, self).__init__()
        self.error = mlp_shuffle_true_SinChannelwise(n_hidden=1024, input_size=4, output_size=4)



# class rational_shuffle_true_SinChannelwise(nn.Module):
#     def __init__(self):
#         super(rational_shuffle_true_SinChannelwise, self).__init__()
#         self.a1 = Parameter(torch.FloatTensor(torch.full((1,1024),-0.1000)))
#         self.a2 = Parameter(torch.FloatTensor(torch.full((1,1024),0.0125)))
#         self.a3 = Parameter(torch.FloatTensor(torch.full((1,1024),0.0001)))
#         self.a4 = Parameter(torch.FloatTensor(torch.full((1,1024),0.001)))
#         self.a5 = Parameter(torch.FloatTensor(torch.full((1,1024),0.1000)))
#     def forward(self, x):
#         # print(x.shape)        # (16384, 1024) (batch size, channels)
#         sig = torch.sigmoid(200*self.a2*(x+self.a3))
#         y = self.a1*sig*(1-sig) + self.a4*torch.sin(200*self.a5*x)
#         return y

# class mlp_shuffle_true_SinChannelwise(nn.Module):
#     def __init__(self, n_hidden, input_size=4, output_size=4):
#         super(mlp_shuffle_true_SinChannelwise, self).__init__()
#         layers = [nn.Linear(input_size, n_hidden),
#                   rational_shuffle_true_SinChannelwise(),
#                   nn.Linear(n_hidden, output_size)]
#         self.nn = nn.Sequential(*layers)
#     def forward(self, p):
#         y = self.nn(p)
#         return y

# class NeurVec_shuffle_true_SinChannelwise(nn.Module):
#     def __init__(self):
#         super(NeurVec_shuffle_true_SinChannelwise, self).__init__()
#         self.error = mlp_shuffle_true_SinChannelwise(n_hidden=1024, input_size=4, output_size=4)



class rational_shuffle_true_SirenChannelwise(nn.Module):
    def __init__(self):
        super(rational_shuffle_true_SirenChannelwise, self).__init__()
    def forward(self, x):
        y = torch.sin(30 * x)
        return y

class mlp_shuffle_true_SirenChannelwise(nn.Module):
    def __init__(self, n_hidden, input_size=4, output_size=4):
        super(mlp_shuffle_true_SirenChannelwise, self).__init__()
        layers = [nn.Linear(input_size, n_hidden),
                  rational_shuffle_true_SirenChannelwise(),
                  nn.Linear(n_hidden, output_size)]
        self.nn = nn.Sequential(*layers)
    def forward(self, p):
        y = self.nn(p)
        return y

class NeurVec_shuffle_true_SirenChannelwise(nn.Module):
    def __init__(self):
        super(NeurVec_shuffle_true_SirenChannelwise, self).__init__()
        self.error = mlp_shuffle_true_SirenChannelwise(n_hidden=1024, input_size=4, output_size=4)



class rational_shuffle_true_SirenChannelwise_wise(nn.Module):
    def __init__(self):
        super(rational_shuffle_true_SirenChannelwise_wise, self).__init__()
        self.a1 = Parameter(torch.FloatTensor(torch.full((1,1024),30.0000)))
    def forward(self, x):
        y = torch.sin(self.a1 * x)
        return y

class mlp_shuffle_true_SirenChannelwise_wise(nn.Module):
    def __init__(self, n_hidden, input_size=4, output_size=4):
        super(mlp_shuffle_true_SirenChannelwise_wise, self).__init__()
        layers = [nn.Linear(input_size, n_hidden),
                  rational_shuffle_true_SirenChannelwise_wise(),
                  nn.Linear(n_hidden, output_size)]
        self.nn = nn.Sequential(*layers)
    def forward(self, p):
        y = self.nn(p)
        return y

class NeurVec_shuffle_true_SirenChannelwise_wise(nn.Module):
    def __init__(self):
        super(NeurVec_shuffle_true_SirenChannelwise_wise, self).__init__()
        self.error = mlp_shuffle_true_SirenChannelwise_wise(n_hidden=1024, input_size=4, output_size=4)




class rational_shuffle_true_gated(nn.Module):
    def __init__(self):
        super(rational_shuffle_true_gated, self).__init__()
        self.a1 = Parameter(torch.FloatTensor(torch.full((1,1024),-0.1000)))
        self.a2 = Parameter(torch.FloatTensor(torch.full((1,1024),2.5)))
        self.a3 = Parameter(torch.FloatTensor(torch.full((1,1024),0.0001)))
        self.a4 = Parameter(torch.FloatTensor(torch.full((1,1024),0.001)))
        self.a5 = Parameter(torch.FloatTensor(torch.full((1,1024),0.1000)))
        self.a6 = Parameter(torch.FloatTensor(torch.full((1,1024),0.5000)))
    def forward(self, x):
        # print(x.shape)        # (16384, 1024) (batch size, channels)
        sig = torch.sigmoid(self.a2*(x+self.a3))
        f1 = self.a1*sig*(1-sig)
        f2 = self.a4*torch.sin(200*self.a5*x)
        wt = torch.sigmoid(self.a6*x)
        y =  wt*f1 + (1-wt)*f2
        return y

class mlp_shuffle_true_gated(nn.Module):
    def __init__(self, n_hidden, input_size=4, output_size=4):
        super(mlp_shuffle_true_gated, self).__init__()
        layers = [nn.Linear(input_size, n_hidden),
                  rational_shuffle_true_gated(),
                  nn.Linear(n_hidden, output_size)]
        self.nn = nn.Sequential(*layers)
    def forward(self, p):
        y = self.nn(p)
        return y

class NeurVec_shuffle_true_gated(nn.Module):
    def __init__(self):
        super(NeurVec_shuffle_true_gated, self).__init__()
        self.error = mlp_shuffle_true_gated(n_hidden=1024, input_size=4, output_size=4)



class rational_shuffle_true_RAFs(nn.Module):
    def __init__(self):
        super(rational_shuffle_true_RAFs, self).__init__()
        # a1 for alpha1, a2 for alpha2, a3 for beta1, a4 for beta2
        self.a1 = Parameter(torch.FloatTensor(torch.full((1,1024),2.0)))
        self.a2 = Parameter(torch.FloatTensor(torch.full((1,1024),1.0)))
        self.a3 = Parameter(torch.FloatTensor(torch.full((1,1024),0.15)))
        self.a4 = Parameter(torch.FloatTensor(torch.full((1,1024),0.03)))
        # 对照实验，保持偏移量
        self.a5 = Parameter(torch.FloatTensor(torch.full((1,1024),0.001)))

    def forward(self, x):
        y = self.a1 * torch.sin(200*self.a3*x) + self.a2 * torch.exp((x-self.a5)*(x-self.a5)/(-2*self.a4*self.a4-0.01))
        return y

class mlp_shuffle_true_RAFs(nn.Module):
    def __init__(self, n_hidden, input_size=4, output_size=4):
        super(mlp_shuffle_true_RAFs, self).__init__()
        layers = [nn.Linear(input_size, n_hidden),
                  rational_shuffle_true_RAFs(),
                  nn.Linear(n_hidden, output_size)]
        self.nn = nn.Sequential(*layers)
        # self.flag = 0
        # self.avg = 0.0
    def forward(self, p):
        # self.flag += 1
        # if self.flag == 500:
        #     a = self.nn[0]
        #     b = self.nn[1]
        #     u, s, v = torch.svd(b(a(p)))
        #     s = s.detach().cpu().numpy()
        #     s = np.sqrt(s)

        #     s = np.sqrt(s)
        #     s_sum = np.sum(s)
        #     s_max = np.max(s, 0)
        #     print(s_sum/s_max)
        #     self.flag = 0
        #     self.avg += s_sum/s_max

        y = self.nn(p)
        return y

class NeurVec_shuffle_true_RAFs(nn.Module):
    def __init__(self):
        super(NeurVec_shuffle_true_RAFs, self).__init__()
        self.error = mlp_shuffle_true_RAFs(n_hidden=1024, input_size=4, output_size=4)


class rational_shuffle_true_gaussian(nn.Module):
    def __init__(self):
        super(rational_shuffle_true_gaussian, self).__init__()
        # a1 for alpha1, a2 for alpha2, a3 for beta1, a4 for beta2
        # self.a1 = Parameter(torch.FloatTensor(torch.full((1,1024),2.0)))
        self.a2 = Parameter(torch.FloatTensor(torch.full((1,1024),1.0)))
        # self.a3 = Parameter(torch.FloatTensor(torch.full((1,1024),0.15)))
        self.a4 = Parameter(torch.FloatTensor(torch.full((1,1024),0.03)))
        # 对照实验，保持偏移量
        self.a5 = Parameter(torch.FloatTensor(torch.full((1,1024),0.001)))

    def forward(self, x):
        y = self.a2 * torch.exp((x-self.a5)*(x-self.a5)/(-2*self.a4*self.a4-0.01))
        return y

class mlp_shuffle_true_gaussian(nn.Module):
    def __init__(self, n_hidden, input_size=4, output_size=4):
        super(mlp_shuffle_true_gaussian, self).__init__()
        layers = [nn.Linear(input_size, n_hidden),
                  rational_shuffle_true_gaussian(),
                  nn.Linear(n_hidden, output_size)]
        self.nn = nn.Sequential(*layers)
        # self.flag = 0
        # self.avg = 0.0
    def forward(self, p):
        # self.flag += 1
        # if self.flag == 500:
        #     a = self.nn[0]
        #     b = self.nn[1]
        #     u, s, v = torch.svd(b(a(p)))
        #     s = s.detach().cpu().numpy()
        #     s = np.sqrt(s)
        #     s_sum = np.sum(s)
        #     s_max = np.max(s, 0)
        #     print(s_sum/s_max)
        #     self.flag = 0
        #     self.avg += s_sum/s_max

        y = self.nn(p)
        return y

class NeurVec_shuffle_true_gaussian(nn.Module):
    def __init__(self):
        super(NeurVec_shuffle_true_gaussian, self).__init__()
        self.error = mlp_shuffle_true_gaussian(n_hidden=1024, input_size=4, output_size=4)



class rational_shuffle_true_SinChannelwise_reparam1(nn.Module):
    def __init__(self):
        super(rational_shuffle_true_SinChannelwise_reparam1, self).__init__()
        # self.a1 = Parameter(torch.FloatTensor(torch.full((1,1024),-0.1000)))
        # self.a2 = Parameter(torch.FloatTensor(torch.full((1,1024),2.5)))
        # self.a3 = Parameter(torch.FloatTensor(torch.full((1,1024),0.0001)))
        # self.a4 = Parameter(torch.FloatTensor(torch.full((1,1024),0.001)))
        # self.a5 = Parameter(torch.FloatTensor(torch.full((1,1024),0.1000)))
        self.a1 = Parameter(torch.FloatTensor(torch.full((1,1024),-0.5)))
        self.a2 = Parameter(torch.FloatTensor(torch.full((1,1024),2.0)))
        self.a3 = Parameter(torch.FloatTensor(torch.full((1,1024),0.0001)))
        self.a4 = Parameter(torch.FloatTensor(torch.full((1,1024),0.5)))
        self.a5 = Parameter(torch.FloatTensor(torch.full((1,1024),0.1000)))
    def forward(self, x):
        # print(x.shape)        # (16384, 1024) (batch size, channels)
        sig = torch.sigmoid(self.a2*(x+self.a3))
        y = self.a1*sig*(1-sig) + self.a4*torch.sin(200*self.a5*x)
        return y

class mlp_shuffle_true_SinChannelwise_reparam1(nn.Module):
    def __init__(self, n_hidden, input_size=4, output_size=4):
        super(mlp_shuffle_true_SinChannelwise_reparam1, self).__init__()
        layers = [nn.Linear(input_size, n_hidden),
                  rational_shuffle_true_SinChannelwise_reparam1(),
                  nn.Linear(n_hidden, output_size)]
        self.nn = nn.Sequential(*layers)
    def forward(self, p):
        y = self.nn(p)
        return y

class NeurVec_shuffle_true_SinChannelwise_reparam1(nn.Module):
    def __init__(self):
        super(NeurVec_shuffle_true_SinChannelwise_reparam1, self).__init__()
        self.error = mlp_shuffle_true_SinChannelwise_reparam1(n_hidden=1024, input_size=4, output_size=4)


class rational_shuffle_true_SinChannelwise_reparam2(nn.Module):
    def __init__(self):
        super(rational_shuffle_true_SinChannelwise_reparam2, self).__init__()
        # self.a1 = Parameter(torch.FloatTensor(torch.full((1,1024),-0.1000)))
        # self.a2 = Parameter(torch.FloatTensor(torch.full((1,1024),2.5)))
        # self.a3 = Parameter(torch.FloatTensor(torch.full((1,1024),0.0001)))
        # self.a4 = Parameter(torch.FloatTensor(torch.full((1,1024),0.001)))
        # self.a5 = Parameter(torch.FloatTensor(torch.full((1,1024),0.1000)))
        self.a1 = Parameter(torch.FloatTensor(torch.full((1,1024),-0.01)))
        self.a2 = Parameter(torch.FloatTensor(torch.full((1,1024),1.0)))
        self.a3 = Parameter(torch.FloatTensor(torch.full((1,1024),0.0001)))
        self.a4 = Parameter(torch.FloatTensor(torch.full((1,1024),0.01)))
        self.a5 = Parameter(torch.FloatTensor(torch.full((1,1024),0.1000)))
    def forward(self, x):
        # print(x.shape)        # (16384, 1024) (batch size, channels)
        sig = torch.sigmoid(self.a2*(x+self.a3))
        y = self.a1*sig*(1-sig) + self.a4*torch.sin(200*self.a5*x)
        return y

class mlp_shuffle_true_SinChannelwise_reparam2(nn.Module):
    def __init__(self, n_hidden, input_size=4, output_size=4):
        super(mlp_shuffle_true_SinChannelwise_reparam2, self).__init__()
        layers = [nn.Linear(input_size, n_hidden),
                  rational_shuffle_true_SinChannelwise_reparam2(),
                  nn.Linear(n_hidden, output_size)]
        self.nn = nn.Sequential(*layers)
    def forward(self, p):
        y = self.nn(p)
        return y

class NeurVec_shuffle_true_SinChannelwise_reparam2(nn.Module):
    def __init__(self):
        super(NeurVec_shuffle_true_SinChannelwise_reparam2, self).__init__()
        self.error = mlp_shuffle_true_SinChannelwise_reparam2(n_hidden=1024, input_size=4, output_size=4)



class rational_shuffle_true_SinChannelwise_reparam3(nn.Module):
    def __init__(self):
        super(rational_shuffle_true_SinChannelwise_reparam3, self).__init__()
        self.a1 = Parameter(torch.FloatTensor(torch.full((1,1024),-0.5)))
        self.a2 = Parameter(torch.FloatTensor(torch.full((1,1024),2.0)))
        self.a3 = Parameter(torch.FloatTensor(torch.full((1,1024),0.0001)))
        self.a4 = Parameter(torch.FloatTensor(torch.full((1,1024),0.5)))
        self.a5 = Parameter(torch.FloatTensor(torch.full((1,1024),0.1000)))
        nn.init.kaiming_normal_(self.a1, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.a2, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.a3, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.a4, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.a5, mode='fan_out', nonlinearity='relu')
        # nn.init.kaiming_normal_(self.a2.data)
        # nn.init.kaiming_normal_(self.a3.data)
        # nn.init.kaiming_normal_(self.a4.data)
        # nn.init.kaiming_normal_(self.a5.data)
    def forward(self, x):
        sig = torch.sigmoid(self.a2*(x+self.a3))
        y = self.a1*sig*(1-sig) + self.a4*torch.sin(200*self.a5*x)
        return y

class mlp_shuffle_true_SinChannelwise_reparam3(nn.Module):
    def __init__(self, n_hidden, input_size=4, output_size=4):
        super(mlp_shuffle_true_SinChannelwise_reparam3, self).__init__()
        layers = [nn.Linear(input_size, n_hidden),
                  rational_shuffle_true_SinChannelwise_reparam3(),
                  nn.Linear(n_hidden, output_size)]
        self.nn = nn.Sequential(*layers)
    def forward(self, p):
        y = self.nn(p)
        return y

class NeurVec_shuffle_true_SinChannelwise_reparam3(nn.Module):
    def __init__(self):
        super(NeurVec_shuffle_true_SinChannelwise_reparam3, self).__init__()
        self.error = mlp_shuffle_true_SinChannelwise_reparam3(n_hidden=1024, input_size=4, output_size=4)



class rational_shuffle_true_SinChannelwise_reparam4(nn.Module):
    def __init__(self):
        super(rational_shuffle_true_SinChannelwise_reparam4, self).__init__()
        self.a1 = Parameter(torch.FloatTensor(torch.full((1,1024),-0.5)))
        self.a2 = Parameter(torch.FloatTensor(torch.full((1,1024),2.0)))
        self.a3 = Parameter(torch.FloatTensor(torch.full((1,1024),0.0001)))
        self.a4 = Parameter(torch.FloatTensor(torch.full((1,1024),0.5)))
        self.a5 = Parameter(torch.FloatTensor(torch.full((1,1024),0.1000)))
        self.a1 = nn.init.normal_(self.a1, mean=-1., std=1.)
        self.a2 = nn.init.normal_(self.a2, mean=2., std=1.)
        self.a3 = nn.init.normal_(self.a3, mean=0.1, std=0.1)
        self.a4 = nn.init.normal_(self.a4, mean=1., std=1)
        self.a5 = nn.init.normal_(self.a5, mean=0.1, std=0.1)
    def forward(self, x):
        sig = torch.sigmoid(self.a2*(x+self.a3))
        y = self.a1*sig*(1-sig) + self.a4*torch.sin(200*self.a5*x)
        return y

class mlp_shuffle_true_SinChannelwise_reparam4(nn.Module):
    def __init__(self, n_hidden, input_size=4, output_size=4):
        super(mlp_shuffle_true_SinChannelwise_reparam4, self).__init__()
        layers = [nn.Linear(input_size, n_hidden),
                  rational_shuffle_true_SinChannelwise_reparam4(),
                  nn.Linear(n_hidden, output_size)]
        self.nn = nn.Sequential(*layers)
    def forward(self, p):
        y = self.nn(p)
        return y

class NeurVec_shuffle_true_SinChannelwise_reparam4(nn.Module):
    def __init__(self):
        super(NeurVec_shuffle_true_SinChannelwise_reparam4, self).__init__()
        self.error = mlp_shuffle_true_SinChannelwise_reparam4(n_hidden=1024, input_size=4, output_size=4)



class rational_shuffle_true_SinChannelwise_bias(nn.Module):
    def __init__(self):
        super(rational_shuffle_true_SinChannelwise_bias, self).__init__()
        self.a1 = Parameter(torch.FloatTensor(torch.full((1,1024),-0.1000)))
        self.a2 = Parameter(torch.FloatTensor(torch.full((1,1024),2.5)))
        self.a3 = Parameter(torch.FloatTensor(torch.full((1,1024),0.0001)))
        self.a4 = Parameter(torch.FloatTensor(torch.full((1,1024),0.001)))
        self.a5 = Parameter(torch.FloatTensor(torch.full((1,1024),0.1000)))
        self.a6 = Parameter(torch.FloatTensor(torch.full((1,1024),0.001)))
    def forward(self, x):
        # print(x.shape)        # (16384, 1024) (batch size, channels)
        sig = torch.sigmoid(self.a2*(x+self.a3))
        y = self.a1*sig*(1-sig) + self.a4*torch.sin(200*self.a5*(x+self.a6))
        return y

class mlp_shuffle_true_SinChannelwise_bias(nn.Module):
    def __init__(self, n_hidden, input_size=4, output_size=4):
        super(mlp_shuffle_true_SinChannelwise_bias, self).__init__()
        layers = [nn.Linear(input_size, n_hidden),
                  rational_shuffle_true_SinChannelwise_bias(),
                  nn.Linear(n_hidden, output_size)]
        self.nn = nn.Sequential(*layers)
        # self.flag = 0
    def forward(self, p):
        y = self.nn(p)
        return y

class NeurVec_shuffle_true_SinChannelwise_bias(nn.Module):
    def __init__(self):
        super(NeurVec_shuffle_true_SinChannelwise_bias, self).__init__()
        self.error = mlp_shuffle_true_SinChannelwise_bias(n_hidden=1024, input_size=4, output_size=4)






class rational_shuffle_true_Siren0_noise0(nn.Module):
    def __init__(self):
        super(rational_shuffle_true_Siren0_noise0, self).__init__()
        self.a1 = Parameter(torch.FloatTensor(torch.full((1,1024),0.10)))
    def forward(self, x):
        y = torch.sin(200*self.a1 * x)
        return y

class mlp_shuffle_true_Siren0_noise0(nn.Module):
    def __init__(self, n_hidden, input_size=4, output_size=4):
        super(mlp_shuffle_true_Siren0_noise0, self).__init__()
        layers = [nn.Linear(input_size, n_hidden),
                  rational_shuffle_true_Siren0_noise0(),
                  nn.Linear(n_hidden, output_size)]
        self.nn = nn.Sequential(*layers)
    def forward(self, p):
        y = self.nn(p)
        return y

class NeurVec_shuffle_true_Siren0_noise0(nn.Module):
    def __init__(self):
        super(NeurVec_shuffle_true_Siren0_noise0, self).__init__()
        self.error = mlp_shuffle_true_Siren0_noise0(n_hidden=1024, input_size=4, output_size=4)




class rational_shuffle_true_Siren1_noise0(nn.Module):
    def __init__(self):
        super(rational_shuffle_true_Siren1_noise0, self).__init__()
        self.a1 = Parameter(torch.FloatTensor(torch.full((1,1024),0.10)))
        self.a2 = Parameter(torch.FloatTensor(torch.full((1,1024),0.628)))
    def forward(self, x):
        y = torch.sin(200*self.a1 * x) + torch.sin( self.a2 * x)
        return y

class mlp_shuffle_true_Siren1_noise0(nn.Module):
    def __init__(self, n_hidden, input_size=4, output_size=4):
        super(mlp_shuffle_true_Siren1_noise0, self).__init__()
        layers = [nn.Linear(input_size, n_hidden),
                  rational_shuffle_true_Siren1_noise0(),
                  nn.Linear(n_hidden, output_size)]
        self.nn = nn.Sequential(*layers)
    def forward(self, p):
        y = self.nn(p)
        return y

class NeurVec_shuffle_true_Siren1_noise0(nn.Module):
    def __init__(self):
        super(NeurVec_shuffle_true_Siren1_noise0, self).__init__()
        self.error = mlp_shuffle_true_Siren1_noise0(n_hidden=1024, input_size=4, output_size=4)




class rational_shuffle_true_Siren2_noise0(nn.Module):
    def __init__(self):
        super(rational_shuffle_true_Siren2_noise0, self).__init__()
        self.a1 = Parameter(torch.FloatTensor(torch.full((1,1024),0.10)))
        self.a2 = Parameter(torch.FloatTensor(torch.full((1,1024),0.628)))
    def forward(self, x):
        y = torch.sin(200*self.a1 * x) + x*torch.sin(100000*self.a2 * x)
        return y

class mlp_shuffle_true_Siren2_noise0(nn.Module):
    def __init__(self, n_hidden, input_size=4, output_size=4):
        super(mlp_shuffle_true_Siren2_noise0, self).__init__()
        layers = [nn.Linear(input_size, n_hidden),
                  rational_shuffle_true_Siren2_noise0(),
                  nn.Linear(n_hidden, output_size)]
        self.nn = nn.Sequential(*layers)
    def forward(self, p):
        y = self.nn(p)
        return y

class NeurVec_shuffle_true_Siren2_noise0(nn.Module):
    def __init__(self):
        super(NeurVec_shuffle_true_Siren2_noise0, self).__init__()
        self.error = mlp_shuffle_true_Siren2_noise0(n_hidden=1024, input_size=4, output_size=4)





class rational_shuffle_true_Siren3_noise0(nn.Module):
    def __init__(self):
        super(rational_shuffle_true_Siren3_noise0, self).__init__()
        self.a1 = Parameter(torch.FloatTensor(torch.full((1,1024),0.10)))
        self.a2 = Parameter(torch.FloatTensor(torch.full((1,1024),0.628)))
        self.a3 = Parameter(torch.FloatTensor(torch.full((1,1024),0.1)))
    def forward(self, x):
        # 1e-5 对应 628000
        y = torch.sin(200*self.a1 * x)*(1+ self.a3*x*torch.sin(1000000*self.a2 * x))
        return y

class mlp_shuffle_true_Siren3_noise0(nn.Module):
    def __init__(self, n_hidden, input_size=4, output_size=4):
        super(mlp_shuffle_true_Siren3_noise0, self).__init__()
        layers = [nn.Linear(input_size, n_hidden),
                  rational_shuffle_true_Siren3_noise0(),
                  nn.Linear(n_hidden, output_size)]
        self.nn = nn.Sequential(*layers)
    def forward(self, p):
        y = self.nn(p)
        return y

class NeurVec_shuffle_true_Siren3_noise0(nn.Module):
    def __init__(self):
        super(NeurVec_shuffle_true_Siren3_noise0, self).__init__()
        self.error = mlp_shuffle_true_Siren3_noise0(n_hidden=1024, input_size=4, output_size=4)





class rational_shuffle_true_SinChannelwise_withlinear(nn.Module):
    def __init__(self):
        super(rational_shuffle_true_SinChannelwise_withlinear, self).__init__()
        self.a1 = Parameter(torch.FloatTensor(torch.full((1,1024),-0.1000)))
        self.a2 = Parameter(torch.FloatTensor(torch.full((1,1024),2.5)))
        self.a3 = Parameter(torch.FloatTensor(torch.full((1,1024),0.0001)))
        self.a4 = Parameter(torch.FloatTensor(torch.full((1,1024),0.001)))
        self.a5 = Parameter(torch.FloatTensor(torch.full((1,1024),0.1000)))
        self.a6 = Parameter(torch.FloatTensor(torch.full((1,1024),0.01)))
        self.a7 = Parameter(torch.FloatTensor(torch.full((1,1024),0.01)))
    def forward(self, x):
        # print(x.shape)        # (16384, 1024) (batch size, channels)
        sig = torch.sigmoid(self.a2*(x+self.a3))
        y = self.a1*sig*(1-sig) + self.a4*torch.sin(200*self.a5*x) + self.a6*x+self.a7
        return y

class mlp_shuffle_true_SinChannelwise_withlinear(nn.Module):
    def __init__(self, n_hidden, input_size=4, output_size=4):
        super(mlp_shuffle_true_SinChannelwise_withlinear, self).__init__()
        layers = [nn.Linear(input_size, n_hidden),
                  rational_shuffle_true_SinChannelwise_withlinear(),
                  nn.Linear(n_hidden, output_size)]
        self.nn = nn.Sequential(*layers)
        # self.flag = 0
        # self.avg = 0.0
    def forward(self, p):
        # self.flag += 1
        # if self.flag == 500:
        #     a = self.nn[0]
        #     b = self.nn[1]
        #     u, s, v = torch.svd(b(a(p)))
        #     s = s.detach().cpu().numpy()
        #     s = np.sqrt(s)

        #     s_sum = np.sum(s)
        #     s_max = np.max(s, 0)
        #     print(s_sum/s_max)
        #     self.flag = 0
        #     self.avg += s_sum/s_max

        y = self.nn(p)
        return y

class NeurVec_shuffle_true_SinChannelwise_withlinear(nn.Module):
    def __init__(self):
        super(NeurVec_shuffle_true_SinChannelwise_withlinear, self).__init__()
        self.error = mlp_shuffle_true_SinChannelwise_withlinear(n_hidden=1024, input_size=4, output_size=4)



# 魔改SinChannelwise之一
class rational_shuffle_true_SinChannelwise_mogai1(nn.Module):
    def __init__(self):
        super(rational_shuffle_true_SinChannelwise_mogai1, self).__init__()
        self.a1 = Parameter(torch.FloatTensor(torch.full((1,1024),-0.1000)))
        self.a2 = Parameter(torch.FloatTensor(torch.full((1,1024),2.5)))
        self.a3 = Parameter(torch.FloatTensor(torch.full((1,1024),0.0001)))
        self.a4 = Parameter(torch.FloatTensor(torch.full((1,1024),0.001)))
        self.a5 = Parameter(torch.FloatTensor(torch.full((1,1024),0.1000)))
        self.a6 = Parameter(torch.FloatTensor(torch.full((1,1024),0.01)))

    def forward(self, x):
        # print(x.shape)        # (16384, 1024) (batch size, channels)
        sig = torch.sigmoid(self.a2*(torch.sin(100*self.a6*x)+self.a3))
        y = self.a1*sig*(1-sig) + self.a4*torch.sin(200*self.a5*x)
        return y

class mlp_shuffle_true_SinChannelwise_mogai1(nn.Module):
    def __init__(self, n_hidden, input_size=4, output_size=4):
        super(mlp_shuffle_true_SinChannelwise_mogai1, self).__init__()
        layers = [nn.Linear(input_size, n_hidden),
                  rational_shuffle_true_SinChannelwise_mogai1(),
                  nn.Linear(n_hidden, output_size)]
        self.nn = nn.Sequential(*layers)
        # self.flag = 0
    def forward(self, p):
        y = self.nn(p)
        return y

class NeurVec_shuffle_true_SinChannelwise_mogai1(nn.Module):
    def __init__(self):
        super(NeurVec_shuffle_true_SinChannelwise_mogai1, self).__init__()
        self.error = mlp_shuffle_true_SinChannelwise_mogai1(n_hidden=1024, input_size=4, output_size=4)



# 魔改SinChannelwise之二
class rational_shuffle_true_SinChannelwise_mogai2(nn.Module):
    def __init__(self):
        super(rational_shuffle_true_SinChannelwise_mogai2, self).__init__()
        self.a1 = Parameter(torch.FloatTensor(torch.full((1,1024),-0.1000)))
        self.a2 = Parameter(torch.FloatTensor(torch.full((1,1024),2.5)))
        self.a3 = Parameter(torch.FloatTensor(torch.full((1,1024),0.0001)))
        self.a6 = Parameter(torch.FloatTensor(torch.full((1,1024),0.01)))
    def forward(self, x):
        # print(x.shape)        # (16384, 1024) (batch size, channels)
        sig = torch.sigmoid(self.a2*(torch.sin(100*self.a6*x)+self.a3))
        y = self.a1*sig*(1-sig)
        return y

class mlp_shuffle_true_SinChannelwise_mogai2(nn.Module):
    def __init__(self, n_hidden, input_size=4, output_size=4):
        super(mlp_shuffle_true_SinChannelwise_mogai2, self).__init__()
        layers = [nn.Linear(input_size, n_hidden),
                  rational_shuffle_true_SinChannelwise_mogai2(),
                  nn.Linear(n_hidden, output_size)]
        self.nn = nn.Sequential(*layers)
        # self.flag = 0
    def forward(self, p):
        y = self.nn(p)
        return y

class NeurVec_shuffle_true_SinChannelwise_mogai2(nn.Module):
    def __init__(self):
        super(NeurVec_shuffle_true_SinChannelwise_mogai2, self).__init__()
        self.error = mlp_shuffle_true_SinChannelwise_mogai2(n_hidden=1024, input_size=4, output_size=4)




# 魔改SinChannelwise之三————非光滑模仿，尖角
class rational_shuffle_true_SinChannelwise_mogai3(nn.Module):
    def __init__(self):
        super(rational_shuffle_true_SinChannelwise_mogai3, self).__init__()
        self.a0 = Parameter(torch.FloatTensor(torch.full((1,1024),0.01)))
        self.a1 = Parameter(torch.FloatTensor(torch.full((1,1024),-0.5)))
        self.a2 = Parameter(torch.FloatTensor(torch.full((1,1024),0.01)))
        self.a3 = Parameter(torch.FloatTensor(torch.full((1,1024),0.01)))
    def forward(self, x):
        # self.a0: 对称轴横坐标
        # self.a1: 斜率，初始为正
        # self.a2: 相对于relu的截距
        # self.a3: relu相对于x轴的移动
        x_tmp0 = x-self.a0
        x_tmp_1 = torch.sign(x_tmp0)*x_tmp0 + self.a2
        # y = self.a1*nn.ReLU(x_tmp_1) - self.a3
        y = self.a1*(torch.sign(x_tmp_1)+1)*x_tmp_1 - self.a3
        return y
    
        # # self.a0, self.a3 : Adaptive weight
        # # self.a1: 曲线最高点
        # # self.a2: 一次函数的交点横坐标，加入self.a2 (理论上要有 self.a2 < self.a1 以保证交点在x轴以上?)
        # # x_tmp0 = torch.min(x+self.a2,-x)
        # # x_tmp1 = torch.max(x_tmp0+self.a1, self.a_zero)
        # x_tmp0 = x + self.a2 + x
        # x_tmp1 = x_tmp0 + self.a1 + self.a_zero
        # y = self.a0 * x_tmp1 + self.a3*torch.sin(200*self.a4*x)

class mlp_shuffle_true_SinChannelwise_mogai3(nn.Module):
    def __init__(self, n_hidden, input_size=4, output_size=4):
        super(mlp_shuffle_true_SinChannelwise_mogai3, self).__init__()
        layers = [nn.Linear(input_size, n_hidden),
                  rational_shuffle_true_SinChannelwise_mogai3(),
                  nn.Linear(n_hidden, output_size)]
        self.nn = nn.Sequential(*layers)
    def forward(self, p):
        y = self.nn(p)
        return y

class NeurVec_shuffle_true_SinChannelwise_mogai3(nn.Module):
    def __init__(self):
        super(NeurVec_shuffle_true_SinChannelwise_mogai3, self).__init__()
        self.error = mlp_shuffle_true_SinChannelwise_mogai3(n_hidden=1024, input_size=4, output_size=4)




# 魔改SinChannelwise之四————光滑模仿
class rational_shuffle_true_RAFsFine(nn.Module):
    def __init__(self):
        super(rational_shuffle_true_RAFsFine, self).__init__()
        ## a1 for alpha1, a2 for alpha2, a3 for beta1, a4 for beta2
        self.a1 = Parameter(torch.FloatTensor(torch.full((1,1024),2.0)))
        self.a2 = Parameter(torch.FloatTensor(torch.full((1,1024),1.0)))
        self.a3 = Parameter(torch.FloatTensor(torch.full((1,1024),0.15)))
        self.a4 = Parameter(torch.FloatTensor(torch.full((1,1024),0.03)))
        self.a5 = Parameter(torch.FloatTensor(torch.full((1,1024),0.001)))
    def forward(self, x):
        y = self.a1 * torch.sin(200*self.a3*x) + self.a2 * torch.exp((x-self.a5)*(x-self.a5)/(-self.a4*self.a4-1e-8))
        return y

class mlp_shuffle_true_RAFsFine(nn.Module):
    def __init__(self, n_hidden, input_size=4, output_size=4):
        super(mlp_shuffle_true_RAFsFine, self).__init__()
        layers = [nn.Linear(input_size, n_hidden),
                  rational_shuffle_true_RAFsFine(),
                  nn.Linear(n_hidden, output_size)]
        self.nn = nn.Sequential(*layers)
    def forward(self, p):
        y = self.nn(p)
        return y

class NeurVec_shuffle_true_RAFsFine(nn.Module):
    def __init__(self):
        super(NeurVec_shuffle_true_RAFsFine, self).__init__()
        self.error = mlp_shuffle_true_RAFsFine(n_hidden=1024, input_size=4, output_size=4)







# 魔改SinChannelwise之五————非光滑模仿，方波
class rational_shuffle_true_SinChannelwise_mogai5(nn.Module):
    def __init__(self):
        super(rational_shuffle_true_SinChannelwise_mogai5, self).__init__()
        self.a0 = Parameter(torch.FloatTensor(torch.full((1,1024),0.001)))
        self.a1 = Parameter(torch.FloatTensor(torch.full((1,1024),-0.25)))
        self.a2 = Parameter(torch.FloatTensor(torch.full((1,1024),-0.001)))
    def forward(self, x):
        # self.a0: 横轴偏置
        # self.a1: 放缩比例
        # self.a2: 纵轴偏置
        x_tmp0 = x-self.a0
        y = self.a1*(torch.sign(x_tmp0)+1) * (torch.sign(1-x_tmp0)+1) + self.a2
        return y

class mlp_shuffle_true_SinChannelwise_mogai5(nn.Module):
    def __init__(self, n_hidden, input_size=4, output_size=4):
        super(mlp_shuffle_true_SinChannelwise_mogai5, self).__init__()
        layers = [nn.Linear(input_size, n_hidden),
                  rational_shuffle_true_SinChannelwise_mogai5(),
                  nn.Linear(n_hidden, output_size)]
        self.nn = nn.Sequential(*layers)
    def forward(self, p):
        y = self.nn(p)
        return y

class NeurVec_shuffle_true_SinChannelwise_mogai5(nn.Module):
    def __init__(self):
        super(NeurVec_shuffle_true_SinChannelwise_mogai5, self).__init__()
        self.error = mlp_shuffle_true_SinChannelwise_mogai5(n_hidden=1024, input_size=4, output_size=4)




class rational_shuffle_true_wholeRAFs(nn.Module):
    def __init__(self):
        super(rational_shuffle_true_wholeRAFs, self).__init__()
        # a1 for alpha1, a2 for alpha2, a3 for beta1, a4 for beta2
        self.a1 = Parameter(torch.FloatTensor(torch.full((1,1024),2.0)))
        self.a2 = Parameter(torch.FloatTensor(torch.full((1,1024),1.0)))
        self.a3 = Parameter(torch.FloatTensor(torch.full((1,1024),0.15)))
        self.a4 = Parameter(torch.FloatTensor(torch.full((1,1024),0.03)))
        # 对照实验，保持偏移量
        self.a5 = Parameter(torch.FloatTensor(torch.full((1,1024),0.001)))
        self.a6 = Parameter(torch.FloatTensor(torch.full((1,1024),0.01)))
        self.a7 = Parameter(torch.FloatTensor(torch.full((1,1024),0.01)))

    def forward(self, x):
        y = self.a1 * torch.sin(200*self.a3*x) + self.a2 * torch.exp((x-self.a5)*(x-self.a5)/(-2*self.a4*self.a4-0.01)) + self.a6*x+self.a7*x**2
        return y

class mlp_shuffle_true_wholeRAFs(nn.Module):
    def __init__(self, n_hidden, input_size=4, output_size=4):
        super(mlp_shuffle_true_wholeRAFs, self).__init__()
        layers = [nn.Linear(input_size, n_hidden),
                  rational_shuffle_true_wholeRAFs(),
                  nn.Linear(n_hidden, output_size)]
        self.nn = nn.Sequential(*layers)
        # self.flag = 0
        # self.avg = 0.0
    def forward(self, p):
        # self.flag += 1
        # if self.flag == 500:
        #     a = self.nn[0]
        #     b = self.nn[1]
        #     u, s, v = torch.svd(b(a(p)))
        #     s = s.detach().cpu().numpy()
        #     s = np.sqrt(s)

        #     s = np.sqrt(s)
        #     s_sum = np.sum(s)
        #     s_max = np.max(s, 0)
        #     print(s_sum/s_max)
        #     self.flag = 0
        #     self.avg += s_sum/s_max

        y = self.nn(p)
        return y

class NeurVec_shuffle_true_wholeRAFs(nn.Module):
    def __init__(self):
        super(NeurVec_shuffle_true_wholeRAFs, self).__init__()
        self.error = mlp_shuffle_true_wholeRAFs(n_hidden=1024, input_size=4, output_size=4)




class rational_shuffle_true_Quadratic(nn.Module):
    def __init__(self):
        super(rational_shuffle_true_Quadratic, self).__init__()
        self.a1 = Parameter(torch.FloatTensor(torch.full((1,1024),-0.1000)))
        self.a2 = Parameter(torch.FloatTensor(torch.full((1,1024),2.5)))
        self.a3 = Parameter(torch.FloatTensor(torch.full((1,1024),0.0001)))
        self.a4 = Parameter(torch.FloatTensor(torch.full((1,1024),0.001)))
        self.a5 = Parameter(torch.FloatTensor(torch.full((1,1024),0.1000)))
        self.a6 = Parameter(torch.FloatTensor(torch.full((1,1024),1.000)))
    def forward(self, x):
        # print(x.shape)        # (16384, 1024) (batch size, channels)
        sig = torch.sigmoid(self.a2*(x+self.a3))
        y = self.a1*sig*(1-sig) + self.a4*torch.sin(200*self.a5*x) + 1/(1+(self.a6*x)**2)
        return y

class mlp_shuffle_true_Quadratic(nn.Module):
    def __init__(self, n_hidden, input_size=4, output_size=4):
        super(mlp_shuffle_true_Quadratic, self).__init__()
        layers = [nn.Linear(input_size, n_hidden),
                  rational_shuffle_true_Quadratic(),
                  nn.Linear(n_hidden, output_size)]
        self.nn = nn.Sequential(*layers)
        # self.flag = 0
        # self.avg = 0.0
    def forward(self, p):
        # self.flag += 1
        # if self.flag == 500:
        #     a = self.nn[0]
        #     b = self.nn[1]
        #     u, s, v = torch.svd(b(a(p)))
        #     s = s.detach().cpu().numpy()
        #     s = np.sqrt(s)

        #     s = np.sqrt(s)
        #     s_sum = np.sum(s)
        #     s_max = np.max(s, 0)
        #     print(s_sum/s_max)
        #     self.flag = 0
        #     self.avg += s_sum/s_max
        y = self.nn(p)
        return y

class NeurVec_shuffle_true_Quadratic(nn.Module):
    def __init__(self):
        super(NeurVec_shuffle_true_Quadratic, self).__init__()
        self.error = mlp_shuffle_true_Quadratic(n_hidden=1024, input_size=4, output_size=4)






class rational_shuffle_true_Laplacian(nn.Module):
    def __init__(self):
        super(rational_shuffle_true_Laplacian, self).__init__()
        self.a1 = Parameter(torch.FloatTensor(torch.full((1,1024),-0.1000)))
        self.a2 = Parameter(torch.FloatTensor(torch.full((1,1024),2.5)))
        self.a3 = Parameter(torch.FloatTensor(torch.full((1,1024),0.0001)))
        self.a4 = Parameter(torch.FloatTensor(torch.full((1,1024),0.001)))
        self.a5 = Parameter(torch.FloatTensor(torch.full((1,1024),0.1000)))
        self.a6 = Parameter(torch.FloatTensor(torch.full((1,1024),1.000)))
    def forward(self, x):
        # print(x.shape)        # (16384, 1024) (batch size, channels)
        sig = torch.sigmoid(self.a2*(x+self.a3))
        y = self.a1*sig*(1-sig) + self.a4*torch.sin(200*self.a5*x) + torch.exp(-torch.abs(x)/self.a6)
        return y

class mlp_shuffle_true_Laplacian(nn.Module):
    def __init__(self, n_hidden, input_size=4, output_size=4):
        super(mlp_shuffle_true_Laplacian, self).__init__()
        layers = [nn.Linear(input_size, n_hidden),
                  rational_shuffle_true_Laplacian(),
                  nn.Linear(n_hidden, output_size)]
        self.nn = nn.Sequential(*layers)
        # self.flag = 0
        # self.avg = 0.0
    def forward(self, p):
        # self.flag += 1
        # if self.flag == 500:
        #     a = self.nn[0]
        #     b = self.nn[1]
        #     u, s, v = torch.svd(b(a(p)))
        #     s = s.detach().cpu().numpy()
        #     s = np.sqrt(s)

        #     s = np.sqrt(s)
        #     s_sum = np.sum(s)
        #     s_max = np.max(s, 0)
        #     print(s_sum/s_max)
        #     self.flag = 0
        #     self.avg += s_sum/s_max
        y = self.nn(p)
        return y

class NeurVec_shuffle_true_Laplacian(nn.Module):
    def __init__(self):
        super(NeurVec_shuffle_true_Laplacian, self).__init__()
        self.error = mlp_shuffle_true_Laplacian(n_hidden=1024, input_size=4, output_size=4)





class rational_shuffle_true_Supergaussian(nn.Module):
    def __init__(self):
        super(rational_shuffle_true_Supergaussian, self).__init__()
        self.a1 = Parameter(torch.FloatTensor(torch.full((1,1024),-0.1000)))
        self.a2 = Parameter(torch.FloatTensor(torch.full((1,1024),2.5)))
        self.a3 = Parameter(torch.FloatTensor(torch.full((1,1024),0.0001)))
        self.a4 = Parameter(torch.FloatTensor(torch.full((1,1024),0.001)))
        self.a5 = Parameter(torch.FloatTensor(torch.full((1,1024),0.1000)))
        self.a6 = Parameter(torch.FloatTensor(torch.full((1,1024),1.000)))
        self.a7 = Parameter(torch.FloatTensor(torch.full((1,1024),1.000)))
    def forward(self, x):
        # print(x.shape)        # (16384, 1024) (batch size, channels)
        sig = torch.sigmoid(self.a2*(x+self.a3))
        y = self.a1*sig*(1-sig) + self.a4*torch.sin(200*self.a5*x) + torch.exp(-x**2/(2*self.a6**2))**self.a7
        return y

class mlp_shuffle_true_Supergaussian(nn.Module):
    def __init__(self, n_hidden, input_size=4, output_size=4):
        super(mlp_shuffle_true_Supergaussian, self).__init__()
        layers = [nn.Linear(input_size, n_hidden),
                  rational_shuffle_true_Supergaussian(),
                  nn.Linear(n_hidden, output_size)]
        self.nn = nn.Sequential(*layers)
        # self.flag = 0
        # self.avg = 0.0
    def forward(self, p):
        # self.flag += 1
        # if self.flag == 500:
        #     a = self.nn[0]
        #     b = self.nn[1]
        #     u, s, v = torch.svd(b(a(p)))
        #     s = s.detach().cpu().numpy()
        #     s = np.sqrt(s)

        #     s = np.sqrt(s)
        #     s_sum = np.sum(s)
        #     s_max = np.max(s, 0)
        #     print(s_sum/s_max)
        #     self.flag = 0
        #     self.avg += s_sum/s_max
        y = self.nn(p)
        return y

class NeurVec_shuffle_true_Supergaussian(nn.Module):
    def __init__(self):
        super(NeurVec_shuffle_true_Supergaussian, self).__init__()
        self.error = mlp_shuffle_true_Supergaussian(n_hidden=1024, input_size=4, output_size=4)



class rational_shuffle_true_Expsin(nn.Module):
    def __init__(self):
        super(rational_shuffle_true_Expsin, self).__init__()
        self.a1 = Parameter(torch.FloatTensor(torch.full((1,1024),-0.1000)))
        self.a2 = Parameter(torch.FloatTensor(torch.full((1,1024),2.5)))
        self.a3 = Parameter(torch.FloatTensor(torch.full((1,1024),0.0001)))
        self.a4 = Parameter(torch.FloatTensor(torch.full((1,1024),0.001)))
        self.a5 = Parameter(torch.FloatTensor(torch.full((1,1024),0.1000)))
        self.a6 = Parameter(torch.FloatTensor(torch.full((1,1024),1.000)))
    def forward(self, x):
        # print(x.shape)        # (16384, 1024) (batch size, channels)
        sig = torch.sigmoid(self.a2*(x+self.a3))
        y = self.a1*sig*(1-sig) + self.a4*torch.sin(200*self.a5*x) + torch.exp(-torch.sin(self.a6*x))
        return y

class mlp_shuffle_true_Expsin(nn.Module):
    def __init__(self, n_hidden, input_size=4, output_size=4):
        super(mlp_shuffle_true_Expsin, self).__init__()
        layers = [nn.Linear(input_size, n_hidden),
                  rational_shuffle_true_Expsin(),
                  nn.Linear(n_hidden, output_size)]
        self.nn = nn.Sequential(*layers)
        # self.flag = 0
        # self.avg = 0.0
    def forward(self, p):
        # self.flag += 1
        # if self.flag == 500:
        #     a = self.nn[0]
        #     b = self.nn[1]
        #     u, s, v = torch.svd(b(a(p)))
        #     s = s.detach().cpu().numpy()
        #     s = np.sqrt(s)

        #     s = np.sqrt(s)
        #     s_sum = np.sum(s)
        #     s_max = np.max(s, 0)
        #     print(s_sum/s_max)
        #     self.flag = 0
        #     self.avg += s_sum/s_max
        y = self.nn(p)
        return y

class NeurVec_shuffle_true_Expsin(nn.Module):
    def __init__(self):
        super(NeurVec_shuffle_true_Expsin, self).__init__()
        self.error = mlp_shuffle_true_Expsin(n_hidden=1024, input_size=4, output_size=4)



class rational_shuffle_true_singleLaplacian(nn.Module):
    def __init__(self):
        super(rational_shuffle_true_singleLaplacian, self).__init__()
        self.a4 = Parameter(torch.FloatTensor(torch.full((1,1024),0.001)))
        self.a5 = Parameter(torch.FloatTensor(torch.full((1,1024),0.1000)))
        self.a6 = Parameter(torch.FloatTensor(torch.full((1,1024),1.000)))
    def forward(self, x):
        # print(x.shape)        # (16384, 1024) (batch size, channels)
        y = self.a4*torch.sin(200*self.a5*x) + torch.exp(-torch.abs(x)/self.a6)
        return y

class mlp_shuffle_true_singleLaplacian(nn.Module):
    def __init__(self, n_hidden, input_size=4, output_size=4):
        super(mlp_shuffle_true_singleLaplacian, self).__init__()
        layers = [nn.Linear(input_size, n_hidden),
                  rational_shuffle_true_singleLaplacian(),
                  nn.Linear(n_hidden, output_size)]
        self.nn = nn.Sequential(*layers)
        # self.flag = 0
        # self.avg = 0.0
    def forward(self, p):
        # self.flag += 1
        # if self.flag == 500:
        #     a = self.nn[0]
        #     b = self.nn[1]
        #     u, s, v = torch.svd(b(a(p)))
        #     s = s.detach().cpu().numpy()
        #     s = np.sqrt(s)

        #     s = np.sqrt(s)
        #     s_sum = np.sum(s)
        #     s_max = np.max(s, 0)
        #     print(s_sum/s_max)
        #     self.flag = 0
        #     self.avg += s_sum/s_max
        y = self.nn(p)
        return y

class NeurVec_shuffle_true_singleLaplacian(nn.Module):
    def __init__(self):
        super(NeurVec_shuffle_true_singleLaplacian, self).__init__()
        self.error = mlp_shuffle_true_singleLaplacian(n_hidden=1024, input_size=4, output_size=4)





class rational_shuffle_true_singleLaplacian_reparam1(nn.Module):
    def __init__(self):
        super(rational_shuffle_true_singleLaplacian_reparam1, self).__init__()
        self.a4 = Parameter(torch.FloatTensor(torch.full((1,1024),0.001)))
        self.a5 = Parameter(torch.FloatTensor(torch.full((1,1024),0.1000)))
        self.a6 = Parameter(torch.FloatTensor(torch.full((1,1024),1.000)))
        self.a7 = Parameter(torch.FloatTensor(torch.full((1,1024),0.001)))
        self.a8 = Parameter(torch.FloatTensor(torch.full((1,1024),0.001)))
    def forward(self, x):
        # print(x.shape)        # (16384, 1024) (batch size, channels)
        y = self.a4*torch.sin(200*self.a5*x) + self.a7*torch.exp(-torch.abs(x-self.a8)/self.a6)
        return y

class mlp_shuffle_true_singleLaplacian_reparam1(nn.Module):
    def __init__(self, n_hidden, input_size=4, output_size=4):
        super(mlp_shuffle_true_singleLaplacian_reparam1, self).__init__()
        layers = [nn.Linear(input_size, n_hidden),
                  rational_shuffle_true_singleLaplacian_reparam1(),
                  nn.Linear(n_hidden, output_size)]
        self.nn = nn.Sequential(*layers)
        # self.flag = 0
        # self.avg = 0.0
    def forward(self, p):
        # self.flag += 1
        # if self.flag == 500:
        #     a = self.nn[0]
        #     b = self.nn[1]
        #     u, s, v = torch.svd(b(a(p)))
        #     s = s.detach().cpu().numpy()
        #     s = np.sqrt(s)

        #     s = np.sqrt(s)
        #     s_sum = np.sum(s)
        #     s_max = np.max(s, 0)
        #     print(s_sum/s_max)
        #     self.flag = 0
        #     self.avg += s_sum/s_max
        y = self.nn(p)
        return y

class NeurVec_shuffle_true_singleLaplacian_reparam1(nn.Module):
    def __init__(self):
        super(NeurVec_shuffle_true_singleLaplacian_reparam1, self).__init__()
        self.error = mlp_shuffle_true_singleLaplacian_reparam1(n_hidden=1024, input_size=4, output_size=4)




class rational_shuffle_true_singleLaplacian_reparam2(nn.Module):
    def __init__(self):
        super(rational_shuffle_true_singleLaplacian_reparam2, self).__init__()
        self.a4 = Parameter(torch.FloatTensor(torch.full((1,1024),0.001)))
        self.a5 = Parameter(torch.FloatTensor(torch.full((1,1024),0.1000)))
        self.a6 = Parameter(torch.FloatTensor(torch.full((1,1024),10.000)))
        self.a7 = Parameter(torch.FloatTensor(torch.full((1,1024),0.001)))
        self.a8 = Parameter(torch.FloatTensor(torch.full((1,1024),0.001)))
    def forward(self, x):
        # print(x.shape)        # (16384, 1024) (batch size, channels)
        y = self.a4*torch.sin(200*self.a5*x) + self.a7*torch.exp(-torch.abs(x-self.a8)/self.a6)
        return y

class mlp_shuffle_true_singleLaplacian_reparam2(nn.Module):
    def __init__(self, n_hidden, input_size=4, output_size=4):
        super(mlp_shuffle_true_singleLaplacian_reparam2, self).__init__()
        layers = [nn.Linear(input_size, n_hidden),
                  rational_shuffle_true_singleLaplacian_reparam2(),
                  nn.Linear(n_hidden, output_size)]
        self.nn = nn.Sequential(*layers)
        # self.flag = 0
        # self.avg = 0.0
    def forward(self, p):
        # self.flag += 1
        # if self.flag == 500:
        #     a = self.nn[0]
        #     b = self.nn[1]
        #     u, s, v = torch.svd(b(a(p)))
        #     s = s.detach().cpu().numpy()
        #     s = np.sqrt(s)

        #     s = np.sqrt(s)
        #     s_sum = np.sum(s)
        #     s_max = np.max(s, 0)
        #     print(s_sum/s_max)
        #     self.flag = 0
        #     self.avg += s_sum/s_max
        y = self.nn(p)
        return y

class NeurVec_shuffle_true_singleLaplacian_reparam2(nn.Module):
    def __init__(self):
        super(NeurVec_shuffle_true_singleLaplacian_reparam2, self).__init__()
        self.error = mlp_shuffle_true_singleLaplacian_reparam2(n_hidden=1024, input_size=4, output_size=4)



class rational_shuffle_true_singleLaplacian_reparam3(nn.Module):
    def __init__(self):
        super(rational_shuffle_true_singleLaplacian_reparam3, self).__init__()
        self.a4 = Parameter(torch.FloatTensor(torch.full((1,1024),0.001)))
        self.a5 = Parameter(torch.FloatTensor(torch.full((1,1024),0.1000)))
        self.a6 = Parameter(torch.FloatTensor(torch.full((1,1024),0.100)))
        self.a7 = Parameter(torch.FloatTensor(torch.full((1,1024),0.001)))
        self.a8 = Parameter(torch.FloatTensor(torch.full((1,1024),0.001)))
    def forward(self, x):
        # print(x.shape)        # (16384, 1024) (batch size, channels)
        y = self.a4*torch.sin(200*self.a5*x) + self.a7*torch.exp(-torch.abs(x-self.a8)/self.a6)
        return y

class mlp_shuffle_true_singleLaplacian_reparam3(nn.Module):
    def __init__(self, n_hidden, input_size=4, output_size=4):
        super(mlp_shuffle_true_singleLaplacian_reparam3, self).__init__()
        layers = [nn.Linear(input_size, n_hidden),
                  rational_shuffle_true_singleLaplacian_reparam3(),
                  nn.Linear(n_hidden, output_size)]
        self.nn = nn.Sequential(*layers)
        # self.flag = 0
        # self.avg = 0.0
    def forward(self, p):
        # self.flag += 1
        # if self.flag == 500:
        #     a = self.nn[0]
        #     b = self.nn[1]
        #     u, s, v = torch.svd(b(a(p)))
        #     s = s.detach().cpu().numpy()
        #     s = np.sqrt(s)

        #     s = np.sqrt(s)
        #     s_sum = np.sum(s)
        #     s_max = np.max(s, 0)
        #     print(s_sum/s_max)
        #     self.flag = 0
        #     self.avg += s_sum/s_max
        y = self.nn(p)
        return y

class NeurVec_shuffle_true_singleLaplacian_reparam3(nn.Module):
    def __init__(self):
        super(NeurVec_shuffle_true_singleLaplacian_reparam3, self).__init__()
        self.error = mlp_shuffle_true_singleLaplacian_reparam3(n_hidden=1024, input_size=4, output_size=4)




class rational_shuffle_true_singleLaplacian_reparam4(nn.Module):
    def __init__(self):
        super(rational_shuffle_true_singleLaplacian_reparam4, self).__init__()
        self.a4 = Parameter(torch.FloatTensor(torch.full((1,1024),0.001)))
        self.a5 = Parameter(torch.FloatTensor(torch.full((1,1024),0.1000)))
        self.a6 = Parameter(torch.FloatTensor(torch.full((1,1024),1.000)))
        self.a7 = Parameter(torch.FloatTensor(torch.full((1,1024),0.001)))
    def forward(self, x):
        # print(x.shape)        # (16384, 1024) (batch size, channels)
        y = self.a4*torch.sin(200*self.a5*x) + self.a7*torch.exp(-torch.abs(x)*10*self.a6)
        return y

class mlp_shuffle_true_singleLaplacian_reparam4(nn.Module):
    def __init__(self, n_hidden, input_size=4, output_size=4):
        super(mlp_shuffle_true_singleLaplacian_reparam4, self).__init__()
        layers = [nn.Linear(input_size, n_hidden),
                  rational_shuffle_true_singleLaplacian_reparam4(),
                  nn.Linear(n_hidden, output_size)]
        self.nn = nn.Sequential(*layers)
        # self.flag = 0
        # self.avg = 0.0
    def forward(self, p):
        # self.flag += 1
        # if self.flag == 500:
        #     a = self.nn[0]
        #     b = self.nn[1]
        #     u, s, v = torch.svd(b(a(p)))
        #     s = s.detach().cpu().numpy()
        #     s = np.sqrt(s)

        #     s = np.sqrt(s)
        #     s_sum = np.sum(s)
        #     s_max = np.max(s, 0)
        #     print(s_sum/s_max)
        #     self.flag = 0
        #     self.avg += s_sum/s_max
        y = self.nn(p)
        return y

class NeurVec_shuffle_true_singleLaplacian_reparam4(nn.Module):
    def __init__(self):
        super(NeurVec_shuffle_true_singleLaplacian_reparam4, self).__init__()
        self.error = mlp_shuffle_true_singleLaplacian_reparam4(n_hidden=1024, input_size=4, output_size=4)


class rational_shuffle_true_channelshareAndnolinearnobias(nn.Module):
    def __init__(self):
        super(rational_shuffle_true_channelshareAndnolinearnobias, self).__init__()
        self.a1 = Parameter(torch.FloatTensor([-0.100]))
        self.a2 = Parameter(torch.FloatTensor([2.5]))
    def forward(self, x):
        # print(x.shape)        # (16384, 1024) (batch size, channels)
        sig = torch.sigmoid(self.a2*x)
        y = self.a1*sig*(1-sig)
        return y


class mlp_shuffle_true_channelshareAndnolinearnobias(nn.Module):
    def __init__(self, n_hidden, input_size=4, output_size=4):
        super(mlp_shuffle_true_channelshareAndnolinearnobias, self).__init__()
        layers = [nn.Linear(input_size, n_hidden),
                  rational_shuffle_true_channelshareAndnolinearnobias(),
                  nn.Linear(n_hidden, output_size)]
        self.nn = nn.Sequential(*layers)
    def forward(self, p):
        y = self.nn(p)
        return y

class NeurVec_shuffle_true_channelshareAndnolinearnobias(nn.Module):
    def __init__(self):
        super(NeurVec_shuffle_true_channelshareAndnolinearnobias, self).__init__()
        self.error = mlp_shuffle_true_channelshareAndnolinearnobias(n_hidden=1024, input_size=4, output_size=4)



































































class rational_shuffle_true_ifnoise(nn.Module):
    def __init__(self):
        super(rational_shuffle_true_ifnoise, self).__init__()
        self.a0 = Parameter(torch.FloatTensor(torch.full((1,1024),0.0140)))
        self.a1 = Parameter(torch.FloatTensor(torch.full((1,1024),-0.1000)))
        self.a2 = Parameter(torch.FloatTensor(torch.full((1,1024),2.5)))
        self.a3 = Parameter(torch.FloatTensor(torch.full((1,1024),0.0070)))
    def forward(self, x):
        sig = torch.sigmoid(self.a2*x)
        y = self.a0*x + self.a1*sig*(1-sig) + self.a3
        return y

class mlp_shuffle_true_ifnoise(nn.Module):
    def __init__(self, n_hidden, input_size=4, output_size=4):
        super(mlp_shuffle_true_ifnoise, self).__init__()
        layers = [nn.Linear(input_size, n_hidden),
                  rational_shuffle_true_ifnoise(),
                  nn.Linear(n_hidden, output_size)]
        self.nn = nn.Sequential(*layers)
        self.sigma = Parameter(torch.FloatTensor([0.001]))
    def forward(self, p):
        noise = torch.FloatTensor(np.random.normal(0, 0.001, p.shape).astype(dtype=np.float32))
        p_noise = p + noise
        p = torch.cat([p,p_noise],dim=1)
        y = self.nn(p)
        return y

class NeurVec_shuffle_true_ifnoise(nn.Module):
    def __init__(self):
        super(NeurVec_shuffle_true_ifnoise, self).__init__()
        self.error = mlp_shuffle_true_ifnoise(n_hidden=1024, input_size=8, output_size=4)











class rational_shuffle_true_jump(nn.Module):
    def __init__(self):
        super(rational_shuffle_true_jump, self).__init__()
        print('rational!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        self.a0 = Parameter(torch.FloatTensor([0.0218]))
        self.a1 = Parameter(torch.FloatTensor([0.5000]))
        self.a2 = Parameter(torch.FloatTensor([1.5957]))
        self.a3 = Parameter(torch.FloatTensor([1.1915]))
        self.b0 = Parameter(torch.FloatTensor([1.0000]))
        self.b1 = Parameter(torch.FloatTensor([0.0000]))
        self.b2 = Parameter(torch.FloatTensor([2.3830]))
    def forward(self, x):
        y = (self.a3*x**3+self.a2*x**2+self.a1*x+self.a0)/(self.b2*x**2+self.b1*x+self.b0)
        return y

class mlp_shuffle_true_jump(nn.Module):
    def __init__(self, n_hidden, input_size=4, output_size=4):
        super(mlp_shuffle_true_jump, self).__init__()
        layers = [nn.Linear(input_size, n_hidden),
                  rational_shuffle_true_jump(),
                  nn.Linear(n_hidden, output_size)]
        self.nn = nn.Sequential(*layers)
    def forward(self, p):
        y = self.nn(p) + p
        return y

class NeurVec_shuffle_true_jump(nn.Module):
    def __init__(self):
        super(NeurVec_shuffle_true_jump, self).__init__()
        self.error = mlp_shuffle_true_jump(n_hidden=1024, input_size=4, output_size=4)


















class rational_morelayer(nn.Module):
    def __init__(self):
        super(rational_morelayer, self).__init__()
        print('rational!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        self.a0 = Parameter(torch.FloatTensor([0.0218]))
        self.a1 = Parameter(torch.FloatTensor([0.5000]))
        self.a2 = Parameter(torch.FloatTensor([1.5957]))
        self.a3 = Parameter(torch.FloatTensor([1.1915]))
        self.b0 = Parameter(torch.FloatTensor([1.0000]))
        self.b1 = Parameter(torch.FloatTensor([0.0000]))
        self.b2 = Parameter(torch.FloatTensor([2.3830]))
    def forward(self, x):
        y = (self.a3*x**3+self.a2*x**2+self.a1*x+self.a0)/(self.b2*x**2+self.b1*x+self.b0)
        return y

class mlp_morelayer(nn.Module):
    def __init__(self, n_hidden, input_size=4, output_size=4):
        super(mlp_morelayer, self).__init__()
        # print(input_size, n_hidden)
        layers = [nn.Linear(input_size, n_hidden),
                  rational_morelayer(),
                  nn.Linear(n_hidden, n_hidden),
                  rational_morelayer(),
                  nn.Linear(n_hidden, output_size)
                  ]
        self.nn = nn.Sequential(*layers)
    def forward(self, p):
        y = self.nn(p)
        return y

class NeurVec_morelayer(nn.Module):
    def __init__(self):
        super(NeurVec_morelayer, self).__init__()
        self.error = mlp_morelayer(n_hidden=512, input_size=4, output_size=4)








class Swish(nn.Module):
	def __init__(self,inplace=True):
		super(Swish,self).__init__()
		self.inplace=inplace
	def forward(self,x):
		if self.inplace:
			x.mul_(torch.sigmoid(x))
			return x
		else:
			return x*torch.sigmoid(x)

class mlp_swish(nn.Module):
    def __init__(self, n_hidden, input_size=4, output_size=4):
        super(mlp_swish, self).__init__()
        # print(input_size, n_hidden)
        layers = [nn.Linear(input_size, n_hidden),
                  Swish(),
                  nn.Linear(n_hidden, output_size),
                  ]
        self.nn = nn.Sequential(*layers)
    def forward(self, p):
        y = self.nn(p)
        return y

class NeurVec_swish(nn.Module):
    def __init__(self):
        super(NeurVec_swish, self).__init__()
        self.error = mlp_swish(n_hidden=1024, input_size=4, output_size=4)








class rational_norm(nn.Module):
    def __init__(self):
        super(rational_norm, self).__init__()
        print('rational!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        self.a0 = Parameter(torch.FloatTensor([0.0218]))
        self.a1 = Parameter(torch.FloatTensor([0.5000]))
        self.a2 = Parameter(torch.FloatTensor([1.5957]))
        self.a3 = Parameter(torch.FloatTensor([1.1915]))
        self.b0 = Parameter(torch.FloatTensor([1.0000]))
        self.b1 = Parameter(torch.FloatTensor([0.0000]))
        self.b2 = Parameter(torch.FloatTensor([2.3830]))
        
    def forward(self, x):
        y = (self.a3*x**3+self.a2*x**2+self.a1*x+self.a0)/(self.b2*x**2+self.b1*x+self.b0)
        return y

class mlp_norm(nn.Module):
    def __init__(self, n_hidden, input_size=4, output_size=4):
        super(mlp_norm, self).__init__()
        # print(input_size, n_hidden)
        layers = [nn.Linear(input_size, n_hidden),
                  rational_norm(),
                  nn.BatchNorm1d(1024),
                  nn.Linear(n_hidden, output_size),
                  ]
        self.nn = nn.Sequential(*layers)
    def forward(self, p):
        y = self.nn(p)
        return y

class NeurVec_norm(nn.Module):
    def __init__(self):
        super(NeurVec_norm, self).__init__()
        self.error = mlp_norm(n_hidden=1024, input_size=4, output_size=4)










class rational_higher_rational(nn.Module):
    def __init__(self):
        super(rational_higher_rational, self).__init__()
        print('rational!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        self.a0 = Parameter(torch.FloatTensor([0.0218]))
        self.a1 = Parameter(torch.FloatTensor([0.5000]))
        self.a2 = Parameter(torch.FloatTensor([1.5957]))
        self.a3 = Parameter(torch.FloatTensor([1.1915]))
        self.a4 = Parameter(torch.FloatTensor([0.0191]))
        self.a5 = Parameter(torch.FloatTensor([0.0019]))
        self.b0 = Parameter(torch.FloatTensor([1.0000]))
        self.b1 = Parameter(torch.FloatTensor([0.0000]))
        self.b2 = Parameter(torch.FloatTensor([2.3830]))
        self.b3 = Parameter(torch.FloatTensor([0.0383]))
        self.b4 = Parameter(torch.FloatTensor([0.0038]))
        
    def forward(self, x):
        y = (self.a5*x**5 + self.a4*x**4 + self.a3*x**3+self.a2*x**2+self.a1*x+self.a0)/(self.b4*x**4 + self.b3*x**3 + self.b2*x**2+self.b1*x+self.b0)
        return y

class mlp_higher_rational(nn.Module):
    def __init__(self, n_hidden, input_size=4, output_size=4):
        super(mlp_higher_rational, self).__init__()
        # print(input_size, n_hidden)
        layers = [nn.Linear(input_size, n_hidden),
                  rational_higher_rational(),
                  nn.Linear(n_hidden, output_size),
                  ]
        self.nn = nn.Sequential(*layers)
    def forward(self, p):
        y = self.nn(p)
        return y

class NeurVec_higher_rational(nn.Module):
    def __init__(self):
        super(NeurVec_higher_rational, self).__init__()
        self.error = mlp_higher_rational(n_hidden=1024, input_size=4, output_size=4)









class rational_b1(nn.Module):
    def __init__(self):
        super(rational_b1, self).__init__()
        print('rational!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        self.a0 = Parameter(torch.FloatTensor([0.0218]))
        self.a1 = Parameter(torch.FloatTensor([0.5000]))
        self.a2 = Parameter(torch.FloatTensor([1.5957]))
        self.a3 = Parameter(torch.FloatTensor([1.1915]))
        self.b0 = Parameter(torch.FloatTensor([1.0000]))
        self.b1 = Parameter(torch.FloatTensor([0.0000]))
        self.b2 = Parameter(torch.FloatTensor([2.3830]))
    def forward(self, x):
        y = (self.a3*x**3+self.a2*x**2+self.a1*x+self.a0)/(self.b2*x**2+self.b1*x+self.b0)
        return y

class mlp_b1(nn.Module):
    def __init__(self, n_hidden, input_size=4, output_size=4):
        super(mlp_b1, self).__init__()
        # print(input_size, n_hidden)
        layers = [nn.Linear(input_size, n_hidden),
                  rational_b1(),
                  nn.Linear(n_hidden, n_hidden),
                  rational_b1(),
                  nn.Linear(n_hidden, output_size)
                  ]
        self.nn = nn.Sequential(*layers)
    def forward(self, p):
        y = self.nn(p)
        return y

class NeurVec_b1(nn.Module):
    def __init__(self):
        super(NeurVec_b1, self).__init__()
        self.error = mlp_b1(n_hidden=1024, input_size=4, output_size=4)









class rational_b2(nn.Module):
    def __init__(self):
        super(rational_b2, self).__init__()
        print('rational!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        self.a0 = Parameter(torch.FloatTensor([0.0218]))
        self.a1 = Parameter(torch.FloatTensor([0.5000]))
        self.a2 = Parameter(torch.FloatTensor([1.5957]))
        self.a3 = Parameter(torch.FloatTensor([1.1915]))
        self.b0 = Parameter(torch.FloatTensor([1.0000]))
        self.b1 = Parameter(torch.FloatTensor([0.0000]))
        self.b2 = Parameter(torch.FloatTensor([2.3830]))
    def forward(self, x):
        y = (self.a3*x**3+self.a2*x**2+self.a1*x+self.a0)/(self.b2*x**2+self.b1*x+self.b0)
        return y

class mlp_b2(nn.Module):
    def __init__(self, n_hidden, input_size=4, output_size=4):
        super(mlp_b2, self).__init__()
        # print(input_size, n_hidden)
        layers = [nn.Linear(input_size, n_hidden),
                  rational_b2(),
                  nn.Linear(n_hidden, n_hidden),
                  rational_b2(),
                  nn.Linear(n_hidden, output_size)
                  ]
        self.nn = nn.Sequential(*layers)
    def forward(self, p):
        y = self.nn(p)
        return y

class NeurVec_b2(nn.Module):
    def __init__(self):
        super(NeurVec_b2, self).__init__()
        self.error = mlp_b2(n_hidden=256, input_size=4, output_size=4)









class rational_b3(nn.Module):
    def __init__(self):
        super(rational_b3, self).__init__()
        print('rational!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        self.a0 = Parameter(torch.FloatTensor([0.0218]))
        self.a1 = Parameter(torch.FloatTensor([0.5000]))
        self.a2 = Parameter(torch.FloatTensor([1.5957]))
        self.a3 = Parameter(torch.FloatTensor([1.1915]))
        self.b0 = Parameter(torch.FloatTensor([1.0000]))
        self.b1 = Parameter(torch.FloatTensor([0.0000]))
        self.b2 = Parameter(torch.FloatTensor([2.3830]))
    def forward(self, x):
        y = (self.a3*x**3+self.a2*x**2+self.a1*x+self.a0)/(self.b2*x**2+self.b1*x+self.b0)
        return y

class mlp_b3(nn.Module):
    def __init__(self, n_hidden, input_size=4, output_size=4):
        super(mlp_b3, self).__init__()
        # print(input_size, n_hidden)
        layers = [nn.Linear(input_size, n_hidden),
                  rational_b3(),
                  nn.Linear(n_hidden, n_hidden),
                  rational_b3(),
                  nn.Linear(n_hidden, output_size)
                  ]
        self.nn = nn.Sequential(*layers)
    def forward(self, p):
        y = self.nn(p)
        return y

class NeurVec_b3(nn.Module):
    def __init__(self):
        super(NeurVec_b3, self).__init__()
        self.error = mlp_b3(n_hidden=64, input_size=4, output_size=4)









class rational_b4(nn.Module):
    def __init__(self):
        super(rational_b4, self).__init__()
        print('rational!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        self.a0 = Parameter(torch.FloatTensor([0.0218]))
        self.a1 = Parameter(torch.FloatTensor([0.5000]))
        self.a2 = Parameter(torch.FloatTensor([1.5957]))
        self.a3 = Parameter(torch.FloatTensor([1.1915]))
        self.b0 = Parameter(torch.FloatTensor([1.0000]))
        self.b1 = Parameter(torch.FloatTensor([0.0000]))
        self.b2 = Parameter(torch.FloatTensor([2.3830]))
    def forward(self, x):
        y = (self.a3*x**3+self.a2*x**2+self.a1*x+self.a0)/(self.b2*x**2+self.b1*x+self.b0)
        return y

class mlp_b4(nn.Module):
    def __init__(self, n_hidden, input_size=4, output_size=4):
        super(mlp_b4, self).__init__()
        # print(input_size, n_hidden)
        layers = [nn.Linear(input_size, n_hidden),
                  rational_b4(),
                  nn.Linear(n_hidden, n_hidden),
                  rational_b4(),
                  nn.Linear(n_hidden, n_hidden),
                  rational_b4(),
                  nn.Linear(n_hidden, output_size)
                  ]
        self.nn = nn.Sequential(*layers)
    def forward(self, p):
        y = self.nn(p)
        return y

class NeurVec_b4(nn.Module):
    def __init__(self):
        super(NeurVec_b4, self).__init__()
        self.error = mlp_b4(n_hidden=512, input_size=4, output_size=4)









class rational_b5(nn.Module):
    def __init__(self):
        super(rational_b5, self).__init__()
        print('rational!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        self.a0 = Parameter(torch.FloatTensor([0.0218]))
        self.a1 = Parameter(torch.FloatTensor([0.5000]))
        self.a2 = Parameter(torch.FloatTensor([1.5957]))
        self.a3 = Parameter(torch.FloatTensor([1.1915]))
        self.b0 = Parameter(torch.FloatTensor([1.0000]))
        self.b1 = Parameter(torch.FloatTensor([0.0000]))
        self.b2 = Parameter(torch.FloatTensor([2.3830]))
    def forward(self, x):
        y = (self.a3*x**3+self.a2*x**2+self.a1*x+self.a0)/(self.b2*x**2+self.b1*x+self.b0)
        return y

class mlp_b5(nn.Module):
    def __init__(self, n_hidden, input_size=4, output_size=4):
        super(mlp_b5, self).__init__()
        # print(input_size, n_hidden)
        layers = [nn.Linear(input_size, n_hidden),
                  rational_b5(),
                  nn.Linear(n_hidden, n_hidden),
                  rational_b5(),
                  nn.Linear(n_hidden, n_hidden),
                  rational_b5(),
                  nn.Linear(n_hidden, output_size)
                  ]
        self.nn = nn.Sequential(*layers)
    def forward(self, p):
        y = self.nn(p)
        return y

class NeurVec_b5(nn.Module):
    def __init__(self):
        super(NeurVec_b5, self).__init__()
        self.error = mlp_b5(n_hidden=256, input_size=4, output_size=4)









class rational_b6(nn.Module):
    def __init__(self):
        super(rational_b6, self).__init__()
        print('rational!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        self.a0 = Parameter(torch.FloatTensor([0.0218]))
        self.a1 = Parameter(torch.FloatTensor([0.5000]))
        self.a2 = Parameter(torch.FloatTensor([1.5957]))
        self.a3 = Parameter(torch.FloatTensor([1.1915]))
        self.b0 = Parameter(torch.FloatTensor([1.0000]))
        self.b1 = Parameter(torch.FloatTensor([0.0000]))
        self.b2 = Parameter(torch.FloatTensor([2.3830]))
        
    def forward(self, x):
        y = (self.a3*x**3+self.a2*x**2+self.a1*x+self.a0)/(self.b2*x**2+self.b1*x+self.b0)
        return y

class mlp_b6(nn.Module):
    def __init__(self, n_hidden, input_size=4, output_size=4):
        super(mlp_b6, self).__init__()
        # print(input_size, n_hidden)
        layers = [nn.Linear(input_size, n_hidden),
                  rational_b6(),
                  nn.Linear(n_hidden, n_hidden),
                  rational_b6(),
                  nn.Linear(n_hidden, n_hidden),
                  rational_b6(),
                  nn.Linear(n_hidden, output_size)
                  ]
        self.nn = nn.Sequential(*layers)
    def forward(self, p):
        y = self.nn(p)
        return y

class NeurVec_b6(nn.Module):
    def __init__(self):
        super(NeurVec_b6, self).__init__()
        self.error = mlp_b6(n_hidden=64, input_size=4, output_size=4)









class rational_b7(nn.Module):
    def __init__(self):
        super(rational_b7, self).__init__()
        print('rational!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        self.a0 = Parameter(torch.FloatTensor([0.0218]))
        self.a1 = Parameter(torch.FloatTensor([0.5000]))
        self.a2 = Parameter(torch.FloatTensor([1.5957]))
        self.a3 = Parameter(torch.FloatTensor([1.1915]))
        self.a4 = Parameter(torch.FloatTensor([0.0191]))
        self.a5 = Parameter(torch.FloatTensor([0.0019]))

        self.b0 = Parameter(torch.FloatTensor([1.0000]))
        self.b1 = Parameter(torch.FloatTensor([0.0000]))
        self.b2 = Parameter(torch.FloatTensor([2.3830]))
        self.b3 = Parameter(torch.FloatTensor([0.0383]))
        self.b4 = Parameter(torch.FloatTensor([0.0038]))
    def forward(self, x):
        y = (self.a5*x**5 + self.a4*x**4 + self.a3*x**3+self.a2*x**2+self.a1*x+self.a0)/(self.b4*x**4 + self.b3*x**3 + self.b2*x**2+self.b1*x+self.b0)
        return y

class mlp_b7(nn.Module):
    def __init__(self, n_hidden, input_size=4, output_size=4):
        super(mlp_b7, self).__init__()
        # print(input_size, n_hidden)
        layers = [nn.Linear(input_size, n_hidden),
                  rational_b7(),
                  nn.Linear(n_hidden, output_size),
                  ]
        self.nn = nn.Sequential(*layers)
    def forward(self, p):
        y = self.nn(p)
        return y

class NeurVec_b7(nn.Module):
    def __init__(self):
        super(NeurVec_b7, self).__init__()
        self.error = mlp_b7(n_hidden=512, input_size=4, output_size=4)









class rational_b8(nn.Module):
    def __init__(self):
        super(rational_b8, self).__init__()
        print('rational!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        self.a0 = Parameter(torch.FloatTensor([0.0218]))
        self.a1 = Parameter(torch.FloatTensor([0.5000]))
        self.a2 = Parameter(torch.FloatTensor([1.5957]))
        self.a3 = Parameter(torch.FloatTensor([1.1915]))
        self.a4 = Parameter(torch.FloatTensor([0.0191]))
        self.a5 = Parameter(torch.FloatTensor([0.0019]))
        self.b0 = Parameter(torch.FloatTensor([1.0000]))
        self.b1 = Parameter(torch.FloatTensor([0.0000]))
        self.b2 = Parameter(torch.FloatTensor([2.3830]))
        self.b3 = Parameter(torch.FloatTensor([0.0383]))
        self.b4 = Parameter(torch.FloatTensor([0.0038]))
    def forward(self, x):
        y = (self.a5*x**5 + self.a4*x**4 + self.a3*x**3+self.a2*x**2+self.a1*x+self.a0)/(self.b4*x**4 + self.b3*x**3 + self.b2*x**2+self.b1*x+self.b0)
        return y

class mlp_b8(nn.Module):
    def __init__(self, n_hidden, input_size=4, output_size=4):
        super(mlp_b8, self).__init__()
        # print(input_size, n_hidden)
        layers = [nn.Linear(input_size, n_hidden),
                  rational_b8(),
                  nn.Linear(n_hidden, output_size)]

        self.nn = nn.Sequential(*layers)
    def forward(self, p):
        y = self.nn(p)
        return y

class NeurVec_b8(nn.Module):
    def __init__(self):
        super(NeurVec_b8, self).__init__()
        self.error = mlp_b8(n_hidden=64, input_size=4, output_size=4)









class rational_b9(nn.Module):
    def __init__(self):
        super(rational_b9, self).__init__()
        print('rational!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        self.a0 = Parameter(torch.FloatTensor([0.0218]))
        self.a1 = Parameter(torch.FloatTensor([0.5000]))
        self.a2 = Parameter(torch.FloatTensor([1.5957]))
        self.a3 = Parameter(torch.FloatTensor([1.1915]))
        self.a4 = Parameter(torch.FloatTensor([0.0191]))
        self.a5 = Parameter(torch.FloatTensor([0.0019]))
        self.a6 = Parameter(torch.FloatTensor([0.0019]))
        self.a7 = Parameter(torch.FloatTensor([0.0019]))

        self.b0 = Parameter(torch.FloatTensor([1.0000]))
        self.b1 = Parameter(torch.FloatTensor([0.0000]))
        self.b2 = Parameter(torch.FloatTensor([2.3830]))
        self.b3 = Parameter(torch.FloatTensor([0.0383]))
        self.b4 = Parameter(torch.FloatTensor([0.0038]))
        self.b5 = Parameter(torch.FloatTensor([0.0038]))
        self.b6 = Parameter(torch.FloatTensor([0.0038]))
    def forward(self, x):
        y = (self.a7*x**7 + self.a6*x**6 + self.a5*x**5 + self.a4*x**4 + self.a3*x**3+self.a2*x**2+self.a1*x+self.a0)/(self.b6*x**6 + self.b5*x**5 + self.b4*x**4 + self.b3*x**3 + self.b2*x**2+self.b1*x+self.b0)
        return y

class mlp_b9(nn.Module):
    def __init__(self, n_hidden, input_size=4, output_size=4):
        super(mlp_b9, self).__init__()
        # print(input_size, n_hidden)
        layers = [nn.Linear(input_size, n_hidden),
                  rational_b9(),
                  nn.Linear(n_hidden, output_size)]
        self.nn = nn.Sequential(*layers)
    def forward(self, p):
        y = self.nn(p)
        return y

class NeurVec_b9(nn.Module):
    def __init__(self):
        super(NeurVec_b9, self).__init__()
        self.error = mlp_b9(n_hidden=1024, input_size=4, output_size=4)









class rational_b10(nn.Module):
    def __init__(self):
        super(rational_b10, self).__init__()
        print('rational!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        self.a0 = Parameter(torch.FloatTensor([0.0218]))
        self.a1 = Parameter(torch.FloatTensor([0.5000]))
        self.a2 = Parameter(torch.FloatTensor([1.5957]))
        self.a3 = Parameter(torch.FloatTensor([1.1915]))
        self.a4 = Parameter(torch.FloatTensor([0.0191]))
        self.a5 = Parameter(torch.FloatTensor([0.0019]))
        self.a6 = Parameter(torch.FloatTensor([0.0019]))
        self.a7 = Parameter(torch.FloatTensor([0.0019]))

        self.b0 = Parameter(torch.FloatTensor([1.0000]))
        self.b1 = Parameter(torch.FloatTensor([0.0000]))
        self.b2 = Parameter(torch.FloatTensor([2.3830]))
        self.b3 = Parameter(torch.FloatTensor([0.0383]))
        self.b4 = Parameter(torch.FloatTensor([0.0038]))
        self.b5 = Parameter(torch.FloatTensor([0.0038]))
        self.b6 = Parameter(torch.FloatTensor([0.0038]))

        
    def forward(self, x):
        y = (self.a7*x**7 + self.a6*x**6 + self.a5*x**5 + self.a4*x**4 + self.a3*x**3+self.a2*x**2+self.a1*x+self.a0)/(self.b6*x**6 + self.b5*x**5 + self.b4*x**4 + self.b3*x**3 + self.b2*x**2+self.b1*x+self.b0)
        return y

class mlp_b10(nn.Module):
    def __init__(self, n_hidden, input_size=4, output_size=4):
        super(mlp_b10, self).__init__()
        # print(input_size, n_hidden)
        layers = [nn.Linear(input_size, n_hidden),
                  rational_b10(),
                  nn.Linear(n_hidden, output_size)]
        self.nn = nn.Sequential(*layers)
    def forward(self, p):
        y = self.nn(p)
        return y

class NeurVec_b10(nn.Module):
    def __init__(self):
        super(NeurVec_b10, self).__init__()
        self.error = mlp_b10(n_hidden=512, input_size=4, output_size=4)










class rational_b11(nn.Module):
    def __init__(self):
        super(rational_b11, self).__init__()
        print('rational!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        self.a0 = Parameter(torch.FloatTensor([0.0218]))
        self.a1 = Parameter(torch.FloatTensor([0.5000]))
        self.a2 = Parameter(torch.FloatTensor([1.5957]))
        self.a3 = Parameter(torch.FloatTensor([1.1915]))
        self.a4 = Parameter(torch.FloatTensor([0.0191]))
        self.a5 = Parameter(torch.FloatTensor([0.0019]))
        self.a6 = Parameter(torch.FloatTensor([0.0019]))
        self.a7 = Parameter(torch.FloatTensor([0.0019]))

        self.b0 = Parameter(torch.FloatTensor([1.0000]))
        self.b1 = Parameter(torch.FloatTensor([0.0000]))
        self.b2 = Parameter(torch.FloatTensor([2.3830]))
        self.b3 = Parameter(torch.FloatTensor([0.0383]))
        self.b4 = Parameter(torch.FloatTensor([0.0038]))
        self.b5 = Parameter(torch.FloatTensor([0.0038]))
        self.b6 = Parameter(torch.FloatTensor([0.0038]))
    def forward(self, x):
        y = (self.a7*x**7 + self.a6*x**6 + self.a5*x**5 + self.a4*x**4 + self.a3*x**3+self.a2*x**2+self.a1*x+self.a0)/(self.b6*x**6 + self.b5*x**5 + self.b4*x**4 + self.b3*x**3 + self.b2*x**2+self.b1*x+self.b0)
        return y


class mlp_b11(nn.Module):
    def __init__(self, n_hidden, input_size=4, output_size=4):
        super(mlp_b11, self).__init__()
        # print(input_size, n_hidden)
        layers = [nn.Linear(input_size, n_hidden),
                  rational_b11(),
                  nn.Linear(n_hidden, output_size)]
        self.nn = nn.Sequential(*layers)
    def forward(self, p):
        y = self.nn(p)
        return y

class NeurVec_b11(nn.Module):
    def __init__(self):
        super(NeurVec_b11, self).__init__()
        self.error = mlp_b11(n_hidden=64, input_size=4, output_size=4)










class rational_b12(nn.Module):
    def __init__(self):
        super(rational_b12, self).__init__()
        print('rational!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        self.a0 = Parameter(torch.FloatTensor([0.0218]))
        self.a1 = Parameter(torch.FloatTensor([0.5000]))
        self.a2 = Parameter(torch.FloatTensor([1.5957]))
        self.b0 = Parameter(torch.FloatTensor([1.0000]))
        self.b1 = Parameter(torch.FloatTensor([1.000]))
        self.b2 = Parameter(torch.FloatTensor([2.3830]))
    def forward(self, x):
        y = (self.a2*x**2+self.a1*x+self.a0)/(self.b2*x**2+self.b1*x+self.b0)
        return y

class mlp_b12(nn.Module):
    def __init__(self, n_hidden, input_size=4, output_size=4):
        super(mlp_b12, self).__init__()
        # print(input_size, n_hidden)
        layers = [nn.Linear(input_size, n_hidden),
                  rational_b12(),
                  nn.Linear(n_hidden, output_size)]
        self.nn = nn.Sequential(*layers)
    def forward(self, p):
        y = self.nn(p)
        return y

class NeurVec_b12(nn.Module):
    def __init__(self):
        super(NeurVec_b12, self).__init__()
        self.error = mlp_b12(n_hidden=1024, input_size=4, output_size=4)










class rational_b13(nn.Module):
    def __init__(self):
        super(rational_b13, self).__init__()
        print('rational!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        self.a0 = Parameter(torch.FloatTensor([0.0218]))
        self.a1 = Parameter(torch.FloatTensor([0.5000]))
        self.a2 = Parameter(torch.FloatTensor([0.5957]))
        self.b0 = Parameter(torch.FloatTensor([1.0000]))
        self.b1 = Parameter(torch.FloatTensor([0.000]))
        self.b2 = Parameter(torch.FloatTensor([2.3830]))
        self.b3 = Parameter(torch.FloatTensor([0.0383]))

        
    def forward(self, x):
        y = (self.a2*x**2+self.a1*x+self.a0)/(self.b3*x**3 + self.b2*x**2+self.b1*x+self.b0)
        return y

class mlp_b13(nn.Module):
    def __init__(self, n_hidden, input_size=4, output_size=4):
        super(mlp_b13, self).__init__()
        # print(input_size, n_hidden)
        layers = [nn.Linear(input_size, n_hidden),
                  rational_b13(),
                  nn.Linear(n_hidden, output_size)]
        self.nn = nn.Sequential(*layers)
    def forward(self, p):
        y = self.nn(p)
        return y

class NeurVec_b13(nn.Module):
    def __init__(self):
        super(NeurVec_b13, self).__init__()
        self.error = mlp_b13(n_hidden=1024, input_size=4, output_size=4)










class rational_b14(nn.Module):
    def __init__(self):
        super(rational_b14, self).__init__()
        print('rational!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        self.a0 = Parameter(torch.FloatTensor([0.0218]))
        self.a1 = Parameter(torch.FloatTensor([0.5000]))
        self.a2 = Parameter(torch.FloatTensor([1.5957]))
        self.a3 = Parameter(torch.FloatTensor([1.1915]))
        self.a4 = Parameter(torch.FloatTensor([1.1915]))

        self.b0 = Parameter(torch.FloatTensor([1.0000]))
        self.b1 = Parameter(torch.FloatTensor([0.0000]))
        self.b2 = Parameter(torch.FloatTensor([2.3830]))
    def forward(self, x):
        y = (self.a4*x**4 + self.a3*x**3+self.a2*x**2+self.a1*x+self.a0)/(self.b2*x**2+self.b1*x+self.b0)
        return y

class mlp_b14(nn.Module):
    def __init__(self, n_hidden, input_size=4, output_size=4):
        super(mlp_b14, self).__init__()
        # print(input_size, n_hidden)
        layers = [nn.Linear(input_size, n_hidden),
                  rational_b14(),
                  nn.Linear(n_hidden, output_size)]
        self.nn = nn.Sequential(*layers)
    def forward(self, p):
        y = self.nn(p)
        return y

class NeurVec_b14(nn.Module):
    def __init__(self):
        super(NeurVec_b14, self).__init__()
        self.error = mlp_b14(n_hidden=1024, input_size=4, output_size=4)










class rational_b15(nn.Module):
    def __init__(self):
        super(rational_b15, self).__init__()
        print('rational!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        self.a0 = Parameter(torch.FloatTensor([0.0218]))
        self.a1 = Parameter(torch.FloatTensor([0.5000]))
        self.a2 = Parameter(torch.FloatTensor([1.5957]))
        self.a3 = Parameter(torch.FloatTensor([1.1915]))
        self.a4 = Parameter(torch.FloatTensor([1.1915]))
        self.a5 = Parameter(torch.FloatTensor([1.1915]))

        self.b0 = Parameter(torch.FloatTensor([1.0000]))
        self.b1 = Parameter(torch.FloatTensor([0.0000]))
        self.b2 = Parameter(torch.FloatTensor([2.3830]))        
    def forward(self, x):
        y = (self.a5*x**5 + self.a4*x**4 + self.a3*x**3+self.a2*x**2+self.a1*x+self.a0)/(self.b2*x**2+self.b1*x+self.b0)
        return y

class mlp_b15(nn.Module):
    def __init__(self, n_hidden, input_size=4, output_size=4):
        super(mlp_b15, self).__init__()
        # print(input_size, n_hidden)
        layers = [nn.Linear(input_size, n_hidden),
                  rational_b15(),
                  nn.Linear(n_hidden, output_size)]
        self.nn = nn.Sequential(*layers)
    def forward(self, p):
        y = self.nn(p)
        return y

class NeurVec_b15(nn.Module):
    def __init__(self):
        super(NeurVec_b15, self).__init__()
        self.error = mlp_b15(n_hidden=1024, input_size=4, output_size=4)











class mlp_b19(nn.Module):
    def __init__(self, n_hidden, input_size=4, output_size=4):
        super(mlp_b19, self).__init__()
        # print(input_size, n_hidden)
        layers = [nn.Linear(input_size, n_hidden),
                  nn.LeakyReLU(),
                  nn.Linear(n_hidden, output_size)]
        self.nn = nn.Sequential(*layers)
    def forward(self, p):
        y = self.nn(p)
        return y

class NeurVec_b19(nn.Module):
    def __init__(self):
        super(NeurVec_b19, self).__init__()
        self.error = mlp_b19(n_hidden=1024, input_size=4, output_size=4)












class mlp_b20(nn.Module):
    def __init__(self, n_hidden, input_size=4, output_size=4):
        super(mlp_b20, self).__init__()
        # print(input_size, n_hidden)
        layers = [nn.Linear(input_size, n_hidden),
                  nn.Softplus(),
                  nn.Linear(n_hidden, output_size)]
        self.nn = nn.Sequential(*layers)
    def forward(self, p):
        y = self.nn(p)
        return y

class NeurVec_b20(nn.Module):
    def __init__(self):
        super(NeurVec_b20, self).__init__()
        self.error = mlp_b20(n_hidden=1024, input_size=4, output_size=4)










class rational_b21(nn.Module):
    def __init__(self):
        super(rational_b21, self).__init__()
        print('rational!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        self.a0 = Parameter(torch.FloatTensor([0.0218]))
        self.a1 = Parameter(torch.FloatTensor([0.5000]))
        self.a2 = Parameter(torch.FloatTensor([1.5957]))
        self.a3 = Parameter(torch.FloatTensor([1.1915]))
        self.b0 = Parameter(torch.FloatTensor([1.0000]))
        self.b1 = Parameter(torch.FloatTensor([0.0000]))
        self.b2 = Parameter(torch.FloatTensor([2.3830]))
        
    def forward(self, x):
        y = (self.a3*x**3+self.a2*x**2+self.a1*x+self.a0)/(self.b2*x**2+self.b1*x+self.b0)
        return y

class mlp_b21(nn.Module):
    def __init__(self, n_hidden, input_size=4, output_size=4):
        super(mlp_b21, self).__init__()
        # print(input_size, n_hidden)
        layers = [nn.Linear(input_size, n_hidden),
                  rational_b21(),
                  nn.Linear(n_hidden, output_size),
                  ]
        self.nn = nn.Sequential(*layers)
    def forward(self, p):
        y = self.nn(p)
        return y

class NeurVec_b21(nn.Module):
    def __init__(self):
        super(NeurVec_b21, self).__init__()
        self.error = mlp_b21(n_hidden=512, input_size=4, output_size=4)












class mlp_b22(nn.Module):
    def __init__(self, n_hidden, input_size=4, output_size=4):
        super(mlp_b22, self).__init__()
        # print(input_size, n_hidden)
        layers = [nn.Linear(input_size, n_hidden),
                  nn.ReLU(),
                  nn.Linear(n_hidden, output_size)
                  ]
        self.nn = nn.Sequential(*layers)
        self.flag = 0
        self.b = None
    def forward(self, p):
        ##
        a = self.nn[0]
        b = a(p).detach().cpu().numpy()
        if self.flag == 0:
            self.b = b
        else:
            self.b = np.concatenate((self.b, b), axis=0)
        self.flag += 1
        # print(b.shape) # (150, 1024)
        # print(self.flag)
        if(self.flag == 500):
            np.save("./before_activate.npy", self.b)
        # print("before activate !!!!!!!!")

        y = self.nn(p)
        return y

class NeurVec_b22(nn.Module):
    def __init__(self):
        super(NeurVec_b22, self).__init__()
        self.error = mlp_b22(n_hidden=1024, input_size=4, output_size=4)
        # self.error = mlp_b22(n_hidden=364, input_size=4, output_size=4) # 359 + 5












class mlp_b23(nn.Module):
    def __init__(self, n_hidden, input_size=4, output_size=4):
        super(mlp_b23, self).__init__()
        # print(input_size, n_hidden)
        layers = [nn.Linear(input_size, n_hidden),
                  nn.ELU(),
                  nn.Linear(n_hidden, output_size)
                  ]

        self.nn = nn.Sequential(*layers)
    def forward(self, p):
        y = self.nn(p)
        return y

class NeurVec_b23(nn.Module):
    def __init__(self):
        super(NeurVec_b23, self).__init__()
        self.error = mlp_b23(n_hidden=1024, input_size=4, output_size=4)











class mlp_b24(nn.Module):
    def __init__(self, n_hidden, input_size=4, output_size=4):
        super(mlp_b24, self).__init__()
        # print(input_size, n_hidden)
        layers = [nn.Linear(input_size, n_hidden),
                  nn.PReLU(),
                  nn.Linear(n_hidden, output_size),
                  ]
        self.nn = nn.Sequential(*layers)
    def forward(self, p):
        y = self.nn(p)
        return y

class NeurVec_b24(nn.Module):
    def __init__(self):
        super(NeurVec_b24, self).__init__()
        self.error = mlp_b24(n_hidden=1024, input_size=4, output_size=4)











class mlp_b25(nn.Module):
    def __init__(self, n_hidden, input_size=4, output_size=4):
        super(mlp_b25, self).__init__()
        # print(input_size, n_hidden)
        layers = [nn.Linear(input_size, n_hidden),
                  nn.SELU(),
                  nn.Linear(n_hidden, output_size),
                  ]
        self.nn = nn.Sequential(*layers)
    def forward(self, p):
        y = self.nn(p)
        return y

class NeurVec_b25(nn.Module):
    def __init__(self):
        super(NeurVec_b25, self).__init__()
        self.error = mlp_b25(n_hidden=1024, input_size=4, output_size=4)












class mlp_b26(nn.Module):
    def __init__(self, n_hidden, input_size=4, output_size=4):
        super(mlp_b26, self).__init__()
        # print(input_size, n_hidden)
        layers = [nn.Linear(input_size, n_hidden),
                  nn.GELU(),
                  nn.Linear(n_hidden, output_size),
                  ]
        self.nn = nn.Sequential(*layers)
    def forward(self, p):
        y = self.nn(p)
        return y

class NeurVec_b26(nn.Module):
    def __init__(self):
        super(NeurVec_b26, self).__init__()
        self.error = mlp_b26(n_hidden=1024, input_size=4, output_size=4)











class mlp_b27(nn.Module):
    def __init__(self, n_hidden, input_size=4, output_size=4):
        super(mlp_b27, self).__init__()
        # print(input_size, n_hidden)
        layers = [nn.Linear(input_size, n_hidden),
                  nn.ReLU6(),
                  nn.Linear(n_hidden, output_size),
                  ]
        self.nn = nn.Sequential(*layers)
    def forward(self, p):
        y = self.nn(p)
        return y

class NeurVec_b27(nn.Module):
    def __init__(self):
        super(NeurVec_b27, self).__init__()
        self.error = mlp_b27(n_hidden=1024, input_size=4, output_size=4)











class mlp_b28(nn.Module):
    def __init__(self, n_hidden, input_size=4, output_size=4):
        super(mlp_b28, self).__init__()
        # print(input_size, n_hidden)
        layers = [nn.Linear(input_size, n_hidden),
                  nn.Mish(),
                  nn.Linear(n_hidden, output_size),
                  ]
        self.nn = nn.Sequential(*layers)
    def forward(self, p):
        y = self.nn(p)
        return y

class NeurVec_b28(nn.Module):
    def __init__(self):
        super(NeurVec_b28, self).__init__()
        self.error = mlp_b28(n_hidden=1024, input_size=4, output_size=4)










class rational_b29(nn.Module):
    def __init__(self):
        super(rational_b29, self).__init__()
        print('rational!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        self.a0 = Parameter(torch.FloatTensor([0.0218]))
        self.a1 = Parameter(torch.FloatTensor([0.5000]))
        self.a2 = Parameter(torch.FloatTensor([1.5957]))
        self.a3 = Parameter(torch.FloatTensor([1.1915]))
        self.b0 = Parameter(torch.FloatTensor([1.0000]))
        self.b1 = Parameter(torch.FloatTensor([0.0000]))
        self.b2 = Parameter(torch.FloatTensor([2.3830]))
        
    def forward(self, x):
        y = (self.a3*x**3+self.a2*x**2+self.a1*x+self.a0)/(self.b2*x**2+self.b1*x+self.b0)
        return y

class mlp_b29(nn.Module):
    def __init__(self, n_hidden, input_size=4, output_size=4):
        super(mlp_b29, self).__init__()
        # print(input_size, n_hidden)
        layers = [nn.Linear(input_size, n_hidden),
                  rational_b29(),
                  nn.Linear(n_hidden, output_size),
                  ]
        self.nn = nn.Sequential(*layers)
    def forward(self, p):
        y = self.nn(p)
        return y

class NeurVec_b29(nn.Module):
    def __init__(self):
        super(NeurVec_b29, self).__init__()
        self.error = mlp_b29(n_hidden=64, input_size=4, output_size=4)










class rational_c1_1(nn.Module):
    def __init__(self):
        super(rational_c1_1, self).__init__()
        print('rational!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        self.a0 = Parameter(torch.FloatTensor([0.0218]))
        self.a1 = Parameter(torch.FloatTensor([0.5000]))
        self.a2 = Parameter(torch.FloatTensor([1.5957]))
        self.a3 = Parameter(torch.FloatTensor([1.1915]))
        self.b0 = Parameter(torch.FloatTensor([1.0000]))
        self.b1 = Parameter(torch.FloatTensor([0.0000]))
        self.b2 = Parameter(torch.FloatTensor([2.3830]))
        
    def forward(self, x):
        y = (self.a3*x**3+self.a2*x**2+self.a1*x+self.a0)/(self.b2*x**2+self.b1*x+self.b0)
        return y

class rational_c1_2(nn.Module):
    def __init__(self):
        super(rational_c1_2, self).__init__()
        print('rational!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        self.a0 = Parameter(torch.FloatTensor([0.0218]))
        self.a1 = Parameter(torch.FloatTensor([0.5000]))
        self.a2 = Parameter(torch.FloatTensor([1.5957]))
        self.a3 = Parameter(torch.FloatTensor([1.1915]))
        self.b0 = Parameter(torch.FloatTensor([1.0000]))
        self.b1 = Parameter(torch.FloatTensor([0.0000]))
        self.b2 = Parameter(torch.FloatTensor([2.3830]))
        
    def forward(self, x):
        y = (self.a3*x**3+self.a2*x**2+self.a1*x+self.a0)/(self.b2*x**2+self.b1*x+self.b0)
        return y

class mlp_c1(nn.Module):
    def __init__(self, n_hidden, input_size=4, output_size=4):
        super(mlp_c1, self).__init__()
        # print(input_size, n_hidden)
        layers = [nn.Linear(int(input_size/2), n_hidden),
                  rational_c1_1(),
                  nn.Linear(n_hidden, int(output_size/2))]
        layers2= [nn.Linear(int(input_size/2), n_hidden),
                  rational_c1_2(),
                  nn.Linear(n_hidden, int(output_size/2))]
        self.nn = nn.Sequential(*layers)
        self.nn2= nn.Sequential(*layers2)
    def forward(self, p):
        # print(p.shape)# [16384, 4]
        y1 = self.nn(p[:,:2])
        y2= self.nn2(p[:,2:])
        y = torch.cat((y1, y2), 1)
        return y

class NeurVec_c1(nn.Module):
    def __init__(self):
        super(NeurVec_c1, self).__init__()
        self.error = mlp_c1(n_hidden=1024, input_size=4, output_size=4)











class rational_c2_1(nn.Module):
    def __init__(self):
        super(rational_c2_1, self).__init__()
        print('rational!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        self.a0 = Parameter(torch.FloatTensor([0.0218]))
        self.a1 = Parameter(torch.FloatTensor([0.5000]))
        self.a2 = Parameter(torch.FloatTensor([1.5957]))
        self.a3 = Parameter(torch.FloatTensor([1.1915]))
        self.b0 = Parameter(torch.FloatTensor([1.0000]))
        self.b1 = Parameter(torch.FloatTensor([0.0000]))
        self.b2 = Parameter(torch.FloatTensor([2.3830]))
        
    def forward(self, x):
        y = (self.a3*x**3+self.a2*x**2+self.a1*x+self.a0)/(self.b2*x**2+self.b1*x+self.b0)
        return y

class rational_c2_2(nn.Module):
    def __init__(self):
        super(rational_c2_2, self).__init__()
        print('rational!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        self.a0 = Parameter(torch.FloatTensor([0.0218]))
        self.a1 = Parameter(torch.FloatTensor([0.5000]))
        self.a2 = Parameter(torch.FloatTensor([1.5957]))
        self.a3 = Parameter(torch.FloatTensor([1.1915]))
        self.b0 = Parameter(torch.FloatTensor([1.0000]))
        self.b1 = Parameter(torch.FloatTensor([0.0000]))
        self.b2 = Parameter(torch.FloatTensor([2.3830]))
        
    def forward(self, x):
        y = (self.a3*x**3+self.a2*x**2+self.a1*x+self.a0)/(self.b2*x**2+self.b1*x+self.b0)
        return y

class mlp_c2(nn.Module):
    def __init__(self, n_hidden, input_size=4, output_size=4):
        super(mlp_c2, self).__init__()
        # print(input_size, n_hidden)
        layers = [nn.Linear(int(input_size/2), n_hidden),
                  rational_c2_1(),
                  nn.Linear(n_hidden, int(output_size/2))]
        layers2= [nn.Linear(int(input_size/2), n_hidden),
                  rational_c2_2(),
                  nn.Linear(n_hidden, int(output_size/2))]
        self.nn = nn.Sequential(*layers)
        self.nn2= nn.Sequential(*layers2)
    def forward(self, p):
        y1 = self.nn(p[:,:2])
        y2= self.nn2(p[:,2:])
        y = torch.cat((y2, y1), 1)
        return y

class NeurVec_c2(nn.Module):
    def __init__(self):
        super(NeurVec_c2, self).__init__()
        self.error = mlp_c2(n_hidden=1024, input_size=4, output_size=4)








class rational_c3_1(nn.Module):
    def __init__(self):
        super(rational_c3_1, self).__init__()
        print('rational!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        self.a0 = Parameter(torch.FloatTensor([0.0218]))
        self.a1 = Parameter(torch.FloatTensor([0.5000]))
        self.a2 = Parameter(torch.FloatTensor([1.5957]))
        self.a3 = Parameter(torch.FloatTensor([1.1915]))

        self.b0 = Parameter(torch.FloatTensor([1.0000]))
        self.b1 = Parameter(torch.FloatTensor([0.0000]))
        self.b2 = Parameter(torch.FloatTensor([2.3830]))

    def forward(self, x):
        y = (self.a3*x**3+self.a2*x**2+self.a1*x+self.a0)/(self.b2*x**2+self.b1*x+self.b0)
        return y
        
class rational_c3_2(nn.Module):
    def __init__(self):
        super(rational_c3_2, self).__init__()
        print('rational!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        self.a0 = Parameter(torch.FloatTensor([0.0218]))
        self.a1 = Parameter(torch.FloatTensor([0.5000]))
        self.a2 = Parameter(torch.FloatTensor([1.5957]))
        self.a3 = Parameter(torch.FloatTensor([1.1915]))

        self.b0 = Parameter(torch.FloatTensor([1.0000]))
        self.b1 = Parameter(torch.FloatTensor([0.0000]))
        self.b2 = Parameter(torch.FloatTensor([2.3830]))

    def forward(self, x):
        y = (self.a3*x**3+self.a2*x**2+self.a1*x+self.a0)/(self.b2*x**2+self.b1*x+self.b0)
        return y

class rational_c3_3(nn.Module):
    def __init__(self):
        super(rational_c3_3, self).__init__()
        print('rational!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        self.a0 = Parameter(torch.FloatTensor([0.0218]))
        self.a1 = Parameter(torch.FloatTensor([0.5000]))
        self.a2 = Parameter(torch.FloatTensor([1.5957]))
        self.a3 = Parameter(torch.FloatTensor([1.1915]))

        self.b0 = Parameter(torch.FloatTensor([1.0000]))
        self.b1 = Parameter(torch.FloatTensor([0.0000]))
        self.b2 = Parameter(torch.FloatTensor([2.3830]))

    def forward(self, x):
        y = (self.a3*x**3+self.a2*x**2+self.a1*x+self.a0) / (self.b2*x**2+self.b1*x+self.b0)
        return y

class mlp_c3(nn.Module):
    def __init__(self, n_hidden, input_size=4, output_size=4):
        super(mlp_c3, self).__init__()
        # print(input_size, n_hidden)
        layers = [nn.Linear(int(input_size/2), n_hidden),
                  rational_c3_1(),
                  nn.Linear(n_hidden, int(output_size/2))]
        layers2= [nn.Linear(int(input_size/2), n_hidden),
                  rational_c3_2(),
                  nn.Linear(n_hidden, int(output_size/2))]        
        layers3= [nn.Linear(output_size, n_hidden),
                  rational_c3_3(),
                  nn.Linear(n_hidden, output_size)]
        self.nn = nn.Sequential(*layers)
        self.nn2= nn.Sequential(*layers2)
        self.nn3= nn.Sequential(*layers3)
    def forward(self, p):
        y1 = self.nn(p[:,:2])
        y2= self.nn2(p[:,2:])
        y = torch.cat((y2, y1), 1)
        y = self.nn3(y)
        return y

class NeurVec_c3(nn.Module):
    def __init__(self):
        super(NeurVec_c3, self).__init__()
        self.error = mlp_c3(n_hidden=1024, input_size=4, output_size=4)











if __name__ == '__main__':
    a = mlp_shuffle_true(40, 40)
    b = torch.randn(10, 40)
    print(a(b).size())