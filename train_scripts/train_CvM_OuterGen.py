'''usage: python -m train_scripts.train_CvM_OuterGen (finds module and runs .py file as script)'''

if __name__ == '__main__':
    
    import os
    import torch
    import torch.optim as optim
    from torch.utils.data import DataLoader
    import numpy as np
    import matplotlib.pyplot as plt

    from main import MixExpPhiStochastic, sampleStochastic, Copula
    from train import load_data, load_data2

    torch.set_default_tensor_type(torch.DoubleTensor)
    
    ##########
    
    train_data, test_data = load_data('./data/synthetic/hac_clayton_dim4.p', 2000, 1000)
    #train_data, test_data = load_data2('./data/realworld/boston_train.p','./data/realworld/boston_test.p')
    #train_data, test_data = load_data2('./data/realworld/intc_msft_train.p','./data/realworld/intc_msft_test.p')
    #train_data, test_data = load_data2('./data/realworld/goog_fb_train.p','./data/realworld/goog_fb_test.p')

    identifier = 'learn_hac_CvM_OuterGen'

    phi = MixExpPhiStochastic()
    net = Copula(phi)
    
    optim_args = \
    {
        'lr': 1e-2,
        'momentum': 0.9
    }

    optimizer = optim.SGD(net.parameters(), optim_args['lr'], optim_args['momentum'])
    num_epochs = 10000
    batch_size = 200
    chkpt_freq = 500
    
    ##########
        
    if not os.path.isdir('./checkpoints/%s' % identifier): 
        os.mkdir('./checkpoints/%s' % identifier)
    if not os.path.isdir('./sample_figs/%s' % identifier):
        os.mkdir('./sample_figs/%s' % identifier)
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=1000000, shuffle=True)
    
    train_loss_per_epoch = []

    for epoch in range(num_epochs):
        loss_per_minibatch = []
        for i, data in enumerate(train_loader, 0):
            optimizer.zero_grad()

            net.phi.resample_M(100)
            
            with torch.no_grad():
                zs = data[:,[0]].numel()
                c12 = torch.zeros(zs)
                c34 = torch.zeros(zs)
                c = torch.zeros(zs)
                for j in range(zs):
                    c12[j] = torch.mean(torch.prod(data[:,[0,1]]<=data[j,[0,1]],axis=1).double())
                    c34[j] = torch.mean(torch.prod(data[:,[2,3]]<=data[j,[2,3]],axis=1).double())
                    c[j] = torch.mean(torch.prod(data[:,[0,1,2,3]]<=data[j,[0,1,2,3]],axis=1).double())

                d = torch.cat([c12.view(-1,1),c34.view(-1,1)],axis=1).detach().double()

            cvmloss = torch.sum(torch.pow(net(d)-c.detach(),2))
            scaleloss = torch.square(torch.mean(net.phi.M)-1)
            
            reg_loss = cvmloss+scaleloss
            reg_loss.backward()
            optimizer.step()
            
            p = net(d, mode='pdf2')

            logloss = -torch.sum(torch.log(p))
            loss_per_minibatch.append((logloss/p.numel()).detach().numpy())

        train_loss_per_epoch.append(np.mean(loss_per_minibatch))

        if epoch % chkpt_freq == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss_per_epoch[-1],
            }, './checkpoints/%s/epoch%s' % (identifier, epoch))

            if True:
                net.phi.resample_M(1000)

                samples = sampleStochastic(net, 2, 1000).detach()
                plt.scatter(samples[:, 0], samples[:, 1])
                plt.savefig('./sample_figs/%s/epoch%s.png' % (identifier, epoch))
                plt.axis("square")
                plt.clf()              

            for i, data in enumerate(test_loader, 0):
                net.zero_grad()
                net.phi.resample_M(1000)
                
                with torch.no_grad():
                    zs = data[:,[0]].numel()
                    c12 = torch.zeros(zs)
                    c34 = torch.zeros(zs)
                    c = torch.zeros(zs)
                    for j in range(zs):
                        c12[j] = torch.mean(torch.prod(data[:,[0,1]]<=data[j,[0,1]],axis=1).double())
                        c34[j] = torch.mean(torch.prod(data[:,[2,3]]<=data[j,[2,3]],axis=1).double())
                        c[j] = torch.mean(torch.prod(data[:,[0,1,2,3]]<=data[j,[0,1,2,3]],axis=1).double())

                d = torch.cat([c12.view(-1,1),c34.view(-1,1)],axis=1).detach().double()
                p = net(d, mode='pdf2')
                logloss = -torch.mean(torch.log(p))

            print('Epoch %s: Train %s, Val %s' %
              (epoch, train_loss_per_epoch[-1], logloss.item()))