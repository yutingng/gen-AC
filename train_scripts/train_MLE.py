'''usage: python -m train_scripts.train_MLE (finds module and runs .py file as script)'''

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
    
    train_data, test_data = load_data('./data/synthetic/clayton_theta5_dim20.p', 2000, 1000)
    #train_data, test_data = load_data('./data/synthetic/clayton_theta5_dim2.p', 2000, 1000)
    #train_data, test_data = load_data2('./data/realworld/boston_train.p','./data/realworld/boston_test.p')
    #train_data, test_data = load_data2('./data/realworld/intc_msft_train.p','./data/realworld/intc_msft_test.p')
    #train_data, test_data = load_data2('./data/realworld/goog_fb_train.p','./data/realworld/goog_fb_test.p')

    identifier = 'learn_clayton_MLE_dim20'

    phi = MixExpPhiStochastic()
    net = Copula(phi)
    
    optim_args = \
    {
        'lr': 1e-5, # it is 1e-3 since torch.sum was used instead of torch.mean for loglikelihood
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

            d = data.detach().clone()
            p = net(d, mode='pdf2')

            scaleloss = torch.square(torch.mean(net.phi.M)-1)
            logloss = -torch.sum(torch.log(p))
            reg_loss = logloss+scaleloss
            reg_loss.backward()

            loss_per_minibatch.append((logloss/p.numel()).detach().numpy())
            optimizer.step()

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
                d = data.detach().clone()
                p = net(d, mode='pdf2')
                logloss = -torch.mean(torch.log(p))

            print('Epoch %s: Train %s, Val %s' %
              (epoch, train_loss_per_epoch[-1], logloss.item()))