
import numpy as np
import torch
import mcubes
from torch import nn
from scipy import spatial
from matplotlib import pyplot as plt
import torch.nn.functional as F

class GeomNet(nn.Module):
    def __init__(self, nlayers, nneurons):
        super(GeomNet, self).__init__()
        ##TODO

    def forward(self,x):
        ##TODO
        return x

#This function computes the gradient of the output with respect to the input (useful for eikonal loss)
def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad


#normal alignment loss
def sdf_loss_align(grad, normals):
    return (1-nn.functional.cosine_similarity(grad, normals, dim = 1)).mean()


def evaluate_loss(geomnet, p, device, lpc, leik, lambda_pc, lambda_eik, batch_size=2000, pc_batch_size=2000):
    pts_random = torch.rand((batch_size, 3), device = device)*2-1
    pts_random.requires_grad = True
  
    sample = torch.randint(p.shape[0], (pc_batch_size,))

    sample_pc = p[sample,0:3]
    sample_nc = p[sample,3:]

    #TODO: get the sdf
    sdf_pc =
    sdf_random =

    #TODO: get the gradient
    grad_pc = 
    grad_random = 
  
    # TODO compute and store the losses
    loss_pc = 
    loss_eik = 
    
    # append all the losses
    lpc.append(float(loss_pc))
    leik.append(float(loss_eik))
  
    # sum the losses
    loss = lambda_pc*loss_pc + lambda_eik*loss_eik

    return loss



def main() :
    p = np.loadtxt('armadillo_sub.xyz')
    
    #TODO compute the bounding box
    
    #TODO normalize the shape (centered at 0, in a cube of size 2)
    p[:,0:3] = 
    
    geomnet = #TODO: create the network
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    geomnet.to(device)
    points = torch.from_numpy(p).float().to(device)
    points.requires_grad = True

    lpc, leik = [], []
    lambda_pc = 1
    lambda_eik = 1

    optim = torch.optim.Adam(params = geomnet.parameters(), lr=1e-3)

    nepochs=5000
    
    for epoch in range(nepochs):
        #TODO one learning step

    #TODO use marching cubes to extract the 0 level set    

    # display the loss
    plt.figure(figsize=(6,4))
    plt.yscale('log')
    plt.plot(lpc, label = 'Point cloud loss ({:.2f})'.format(lpc[-1]))
    plt.plot(leik, label = 'Eikonal loss ({:.2f})'.format(leik[-1]))
    plt.xlabel("Epochs")
    plt.legend()
    plt.savefig("loss.pdf")
    plt.close()

main()
        
