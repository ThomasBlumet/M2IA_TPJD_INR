import numpy as np
import torch
import mcubes
from torch import nn
from scipy import spatial
from matplotlib import pyplot as plt
import torch.nn.functional as F
import torchvision.transforms as transforms 

from PIL import Image 

class ReLuNet(nn.Module):
    def __init__(self, ninputchannels):
        super(ReLuNet, self).__init__()
        #TODO


    def forward(self,x):
        #TODO
    


def evaluate_loss(relunet, pts_gt, sdf_gt, device, lpc, batch_size=2000, delta = 0.1, pc_batch_size=2000):
    #pts_random = torch.rand((batch_size, 3), device = device)*2-1
    indices = torch.randint(pts_gt.shape[0], (batch_size,))
    pts_gt_sub = pts_gt[indices,:]

    #TODO: compute the result

    # compute and store the losses
    loss = #TODO

    # append all the losses
    lpc.append(float(loss.item()))
  
    return loss



def main_shape() :
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    p = np.loadtxt('armadillo_sub.xyz')
    
    #compute the enclosing grid
    maxx = np.max(p[:,0])
    minx = np.min(p[:,0])
    maxy = np.max(p[:,1])
    miny = np.min(p[:,1])
    maxz = np.max(p[:,2])
    minz = np.min(p[:,2])
    
    #normalize the shape 
    maxdim = np.max((maxx-minx, maxy-miny, maxz-minz))
    
    p[:,0:3] = 1.9999*(p[:,0:3] - [minx,miny,minz])/maxdim-0.99999

    #preparing gt points:
    #TODO: using a kdtree find the groundtruth distance from the points to the shape
    gtsdf = 
    
    geomnet = 

    geomnet.to(device)
    gtpoints = torch.from_numpy(gtp).float().to(device)
    gtsdf = torch.from_numpy(sdf).float().to(device)

    lpc = []

    optim = torch.optim.Adam(params = geomnet.parameters(), lr=1e-5)

    nepochs=10000
    
    for epoch in range(nepochs):
        #TODO do one step training

        if epoch % 100 == 0:
            print(f"Epoch {epoch}/{nepochs} - loss : {loss.item()}")

    #use marching cubes to extract the shape
    

    # display the result
    plt.figure(figsize=(6,4))
    plt.yscale('log')
    plt.plot(lpc, label = 'Point cloud loss ({:.2f})'.format(lpc[-1]))
    plt.xlabel("Epochs")
    plt.legend()
    plt.savefig("loss.pdf")
    plt.close()


main_shape()
