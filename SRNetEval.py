import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.nn.functional import upsample

def vis_vf(U,V,P,title=None,ax=None):
    lx = 1.      # width of box
    ly = 1.      # height of box
    nx,ny = U.shape
    
    q = max(int(nx/20),1)

    
    x = np.linspace(0,lx,nx+1); x = (x[1:] + x[:-1])/2.
    y = np.linspace(0,ly,ny+1); y = (y[1:] + y[:-1])/2.
    [X, Y] = np.meshgrid(x, y)
    #P = P.T
    #U = avg(np.vstack([uW, U, uE]), axis=0).T
    #V = avg(np.hstack([vS, V, vN]), axis=1).T
    if not ax:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
    # plotting the pressure field as a contour
    cb = ax.contourf(X, Y, P, alpha=0.5, cmap='viridis')  
    plt.colorbar(cb)
    # plotting the pressure field outlines
    ax.contour(X, Y, P, cmap='viridis')  
    # plotting velocity field
    ax.quiver(X[::q, ::q], Y[::q, ::q], U[::q, ::q], V[::q, ::q])
    plt.xlabel('X')
    plt.ylabel('Y')
    if title:
        plt.title(title)
    else:
        plt.title('Flow Field')

def bil_img(img,sf=2):
    return upsample(torch.Tensor(img).unsqueeze(0),scale_factor=sf,mode='bilinear')[0,...].numpy()

def mse(a,b,weights=None,channel=False):
    if weights is None:
        weights = np.ones(a.shape[0])
    if channel:
        return np.multiply(np.power(a-b,2).mean(axis=(-2,-1)),weights)
    return np.dot(np.power(a-b,2).mean(axis=(-2,-1)),weights)/a.shape[0]

def norm_mse(img,gt,weights=None,channel=False):
    if weights is None:
        weights = np.ones(img.shape[0])
    errors = mse(img,gt,weights,channel=True)/mse(gt,np.zeros_like(gt),weights,channel=True)
    if channel:
        return np.sqrt(errors)
    else:
        return np.sqrt(np.mean(errors))

def vec_comps(img):
    U,V,P = img
    # magnitude
    mag = np.sqrt(np.power(U,2)+np.power(V,2))
    
    # angle
    ang = np.arctan2(V,U)
    return np.array([mag, ang, P])
                    
def visualize_channels(imgs, imgnames,channelnames=['U','V','P']):
    for i,c in enumerate(channelnames):
        plt.figure(figsize=(len(imgnames)*4,4))
        for j,imgname in enumerate(imgnames):
            plt.subplot(1,len(imgnames),j+1)
            plt.title(imgname + ' ' + c)
            plt.imshow(imgs[j,i])
            plt.colorbar()
            
    
def display_results(net,netdata,lowres_data,highres_data=None,GPU=False,idx=None):
    idx = idx if idx is not None else np.random.randint(0,len(netdata))
    print('idx',idx)
    sample = netdata[idx]
    data,label = sample
    data = data.unsqueeze(0)
    
    # Pass through net
    if GPU:
        output = net(data.cuda())[0,...].cpu().detach().numpy()
    else:
        output = net(data)[0,...].detach().numpy()
    
    # Grab lowres
    lowres = lowres_data[idx]
    
    # Fix bilinear
    bil = data[0,:-1,...].numpy()
    
    # fix netout
    netout = output + bil
    
    # Create GT
    gt = label.numpy() + bil
        
    if highres_data is not None:
        # need to upsample netout
        gt = highres_data[idx]
        sf = int(gt.shape[-1]/lowres.shape[-1])
        bil = bil_img(lowres,sf)
        netout = bil_img(netout,int(gt.shape[-1]/netout.shape[-1]))

    # Calculate Error
    print('Bilinear Error: %.4f, %.4f ' %(mse(bil,gt), norm_mse(bil,gt)))
    print('Net Error: %.4f, %.4f ' %(mse(netout,gt), norm_mse(netout,gt)))
    
    # Display Raw channels
    imgnames = ['Bilinear','GT','Netout']
    visualize_channels(np.array([bil,gt,netout]),imgnames=imgnames)
    
    # Display Vector Channels
    vec_imgs = np.array([vec_comps(img)[:2] for img in [bil,gt,netout]])
    visualize_channels(vec_imgs,imgnames=imgnames,channelnames=['Flow Mag','Flow Angle'])
    
    # Display Flows
    fig = plt.figure(figsize=(15,15))
    U,V,P = netout
    ax = fig.add_subplot(2,2,1)
    vis_vf(U,V,P,title='Net Flow',ax=ax)
    
    U,V,P = gt
    ax = fig.add_subplot(2,2,2)
    vis_vf(U,V,P,title='Ground Truth',ax=ax)
    
    U,V,P = bil
    ax = fig.add_subplot(2,2,3)
    vis_vf(U,V,P,title='Bilinear',ax=ax)
    
    U,V,P = lowres
    ax = fig.add_subplot(2,2,4)
    vis_vf(U,V,P,title='Low Res Flow',ax=ax)

def compute_error(net,testset,lowres_data,highres_data=None,GPU=False,vec_metric=False):
    n = len(testset)
    #mses = np.zeros((n,3))
    #vec_mses = np.zeros((n,3))
    #bil_mses = np.zeros((n,3))
    nmses = np.zeros((n,3))
    nvec_mses = np.zeros((n,3))
    nbilvec_mses = np.zeros((n,3))
    nbil_mses = np.zeros((n,3))
    for idx in range(n):
        sample = testset[idx]
        data,label = sample
        data = data.unsqueeze(0)

        # Pass through net
        if GPU:
            output = net(data.cuda())[0,...].cpu().detach().numpy()
        else:
            output = net(data)[0,...].detach().numpy()

        # Grab lowres
        lowres = lowres_data[idx]
        
        # Fix bilinear
        bil = data[0,:-1,...].numpy()

        # fix netout
        netout = output + bil

        # Create GT
        gt = label.numpy() + bil

        if highres_data is not None and bil.shape[-1] != highres_data[idx].shape[-1]:
            # need to upsample netout
            gt = highres_data[idx]
            sf = int(gt.shape[-1]/lowres.shape[-1])
            bil = bil_img(lowres,sf)
            netout = bil_img(netout,int(gt.shape[-1]/netout.shape[-1]))


        nmses[idx] = norm_mse(netout,gt,channel=True)

        # Compare against bil
        nbil_mses[idx] = norm_mse(bil,gt,channel=True)
        
        # Compare Vector Scores
        if vec_metric:
            gt_vec = vec_comps(gt)
            bil_vec = vec_comps(bil)
            netout_vec = vec_comps(netout)
            weights = np.ones(3)
            weights[1] /= np.pi
    
        
            nvec_mses[idx] = norm_mse(netout_vec,gt_vec,weights=weights,channel=True)
            nbilvec_mses[idx] = norm_mse(bil_vec,gt_vec,weights=weights,channel=True)
        
    print('Channel\tNet Error\tBilinear\tVec Net\tVec Bil')
    for i,c in enumerate(['U','V','P']):
        print(c + '\t%.3f\t%.3f\t%.3f\t%.3f' % (nmses[:,i].mean(), nbil_mses[:,i].mean(),nvec_mses[:,i].mean(), nbilvec_mses[:,i].mean()))

    figwidth = 2 if vec_metric else 1 
    fig = plt.figure(figsize=(12,8))
    channels = ['U','V','P']
    for i in range(3):
        plt.subplot(4,figwidth,figwidth*i+1)
        plt.plot(nmses[:,i],label='Net error')
        plt.plot(nbil_mses[:,i],label='Bilinear Error')
        plt.title(channels[i])
        plt.legend(loc=0)
        
        if vec_metric:
            plt.subplot(4,2,2*i+2)
            plt.plot(nvec_mses[:,i],label='Net error')
            plt.plot(nbilvec_mses[:,i],label='Bilinear Error')
            plt.title('Vector ' +channels[i])
            plt.legend(loc=0)
    fig.tight_layout()


    plt.show()
    
    if vec_metric:
        return nvec_mses, nbilvec_mses
    return nmses, nbil_mses

