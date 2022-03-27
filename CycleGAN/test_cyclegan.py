
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from torchvision import datasets, models, transforms
#from model import *
import os
import torch
from torch import cdist
from torch.autograd import Variable
from skimage import io
from scipy import fftpack
import numpy as np
from torch import nn
import datetime
import encoder1_1
import encoder2_1
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from sklearn import metrics

import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html

#################################################################################################################
# HYPER PARAMETERS INITIALIZING
parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
parser.add_argument('--data_train',default='./data/train',help='root directory for training data')
parser.add_argument('--data_test',default='./data/test',help='root directory for testing data')
parser.add_argument('--seed', default=1, type=int, help='manual seed')
parser.add_argument('--batch_size', default=4, type=int, help='batch size')
parser.add_argument('--set_size', default=4, type=int, help='set size')
parser.add_argument('--savedir', default='/mnt/scratch/asnanivi/runs')
parser.add_argument('--model_dir', default='./models')
parser.add_argument('--template_strength', default=0.3, type=float, help='set size')
parser.add_argument('--image_size', default=128, type=int, help='set size')

opt1 = parser.parse_args()
print(opt1)
print("Random Seed: ", opt1.seed)

size=opt1.image_size
set_size=opt1.set_size
b_s=opt1.batch_size
m=opt1.template_strength


opt = TestOptions().parse()  # get test options
# hard-code some parameters for test
opt.num_threads = 0   # test code only supports num_threads = 0
opt.batch_size = 1    # test code only supports batch_size = 1
opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
model = create_model(opt)      # create a model given opt.model and other options
model.setup(opt)               # regular setup: load and print networks; create schedulers

if opt.eval:
    model.eval()
        
device=torch.device("cuda:0")
torch.backends.deterministic = True
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)

sig = str(datetime.datetime.now())

size=128
set_size=3
b_s=1


class vector_var(nn.Module):
    def __init__(self , set_size, size):
        super(vector_var, self).__init__()
        A = torch.rand(set_size,1,size,size, device='cpu')
        self.A = nn.Parameter(A)
        
    def forward(self):
        return self.A

        
        
def roll_n(X, axis, n):
    f_idx = tuple(slice(None, None, None) if i != axis else slice(0,n,None) 
                  for i in range(X.dim()))
    b_idx = tuple(slice(None, None, None) if i != axis else slice(n,None,None)
                  for i in range(X.dim()))
    #print(axis,n,f_idx,b_idx)
    front = X[f_idx]
    back = X[b_idx]
    return torch.cat([back, front],axis)

def fftshift(real, imag):
    for dim in range(1, len(real.size())):
        real = roll_n(real, axis=dim, n=real.size(dim)//2)
        imag = roll_n(imag, axis=dim, n=imag.size(dim)//2)
    return real, imag
    
    
train_path=opt1.data_train
test_path=opt1.data_test
save_dir=opt1.savedir
os.makedirs('%s/logs/%s' % (save_dir, sig), exist_ok=True)
os.makedirs('%s/result_5/%s' % (save_dir, sig), exist_ok=True)

best_acc = 0
start_epoch = 1

transform_train = transforms.Compose([
#transforms.Resize((size,size)),
#transforms.RandomResizedCrop(227),
transforms.ToTensor(),
transforms.Normalize((0.6490, 0.6490, 0.6490), (0.1269, 0.1269, 0.1269))
])




dataset_train = datasets.ImageFolder(train_path, transform_train)
dataset_test = datasets.ImageFolder(test_path, transform_train)
print(len(dataset_train))
print(len(dataset_test))
train_loader = torch.utils.data.DataLoader(dataset_train,batch_size=b_s,shuffle =True, num_workers=1)
test_loader = torch.utils.data.DataLoader(dataset_test,batch_size=b_s,shuffle =True, num_workers=1)



encoder2_model=encoder2_1.encoder2().to(device)  
optimizer_2 = torch.optim.Adam(encoder2_model.parameters(), lr=0.0001)





signal_set_init=vector_var(set_size, size).cuda()
optimizer_3 = torch.optim.Adam(signal_set_init.parameters(), lr=0.0001)

signal_set=signal_set_init()



l2=torch.nn.MSELoss().to(device)
l_c=torch.nn.CrossEntropyLoss().to(device)
l_pair=torch.nn.PairwiseDistance(p=2.0, eps=1e-06, keepdim=False)
cos = nn.CosineSimilarity(dim=1, eps=1e-6)
cos1 = nn.CosineSimilarity(dim=0, eps=1e-6)
state = {
    
    'state_dict_encoder2':encoder2_model.state_dict(),
    'optimizer_2': optimizer_2.state_dict(),
    'state_dict_signal_set':signal_set_init.state_dict(),
    'optimizer_3': optimizer_3.state_dict()
    
}


state1 = torch.load(opt1.mode_dir+'/10_model.pickle')
encoder2_model.load_state_dict(state1['state_dict_encoder2'])
optimizer_2.load_state_dict(state1['optimizer_2'])
signal_set_init.load_state_dict(state1['state_dict_signal_set'])
optimizer_3.load_state_dict(state1['optimizer_3'])






def test(data):
   
    
   
    with torch.no_grad():
        
        signal_sel=torch.randint(0,set_size,(b_s,))
        print(signal_sel)
        signal_est=signal_set[signal_sel.type(torch.cuda.LongTensor),:]


        signal_est_red=0.08*signal_est.clone() 
        signal_set_red=0.08*signal_set.clone() 
        
        
        signal_est_red_resize=F.interpolate(signal_est_red, size=256)
        print(signal_est_red_resize.shape)
        input_with_signal=model.set_input(data, signal_est_red_resize.detach().cpu())  # unpack data from data loader
        model.test()           # run inference
        gen_img = model.get_current_visuals()  # get image results
        gen_img_with_signal=gen_img["fake"]
        print(gen_img_with_signal.shape)
        input_with_signal_resize=F.interpolate(input_with_signal, size=128)
        gen_img_with_signal_resize=F.interpolate(gen_img_with_signal, size=128)
        
        
        signal_rec=encoder2_model(input_with_signal_resize)
        

        signal_fake=encoder2_model(gen_img_with_signal_resize )

        
        dist_fake=torch.zeros([b_s,set_size,2], dtype=torch.float32).to(device)
        for i in range(set_size):
            dist_fake[:,i,0]=cos(signal_set_red[i,:].reshape( -1).unsqueeze(0), signal_fake.reshape(signal_fake.size(0), -1))
            dist_fake[:,i,1]=cos(signal_set_red[i,:].reshape( -1).unsqueeze(0), signal_rec.reshape(signal_fake.size(0), -1))
    
    return  signal_set,data["A"].type(torch.cuda.FloatTensor) ,input_with_signal, signal_rec,gen_img_with_signal.permute(0,3,1,2).type(torch.cuda.FloatTensor) ,signal_fake, dist_fake


epochs=5

for epoch in range(epochs):
    all_y=[]
    all_y_test=[]
    flag=0
    flag1=0
    count=0
    itr=0
    
    for batch_idx_test, data in enumerate(dataset):
    
        signal_set,input_images ,input_with_signal, signal_rec,gen_img_with_signal,signal_fake, dist_fake=test(data)
        #all_y_test.append(np.asarray(labels_test))
        if flag1==0:
            all_dist_fake=dist_fake.detach()
            
            all_signal_set=signal_set.detach()
            all_input_images=input_images.detach()
            all_input_with_signal=input_with_signal.detach()
            all_signal_rec=signal_rec.detach()
            all_gen_img_with_signal=gen_img_with_signal.detach()
            all_signal_fake=signal_fake.detach()
            flag1=1
            #break
        else:
            all_dist_fake=torch.cat([all_dist_fake,dist_fake.detach()], dim=0)
            if count>161500:
                
                all_signal_set=torch.cat([all_signal_set,signal_set.detach()], dim=0)
                all_input_images=torch.cat([all_input_images,input_images.detach()], dim=0)
                all_input_with_signal=torch.cat([all_input_with_signal,input_with_signal.detach()], dim=0)
                all_signal_rec=torch.cat([all_signal_rec,signal_rec.detach()], dim=0)
                all_gen_img_with_signal=torch.cat([all_gen_img_with_signal,gen_img_with_signal.detach()], dim=0)
                all_signal_fake=torch.cat([all_signal_fake,signal_fake.detach()], dim=0)
    
    print(epoch)     
    torch.save(all_dist_fake, '%s/result_5/%s/all_dist_fake_%d.pickle' % (save_dir, sig, epoch))
    
    torch.save(all_signal_set, '%s/result_5/%s/all_signal_set_%d.pickle' % (save_dir, sig, epoch))
    torch.save(all_input_images, '%s/result_5/%s/all_input_images_%d.pickle' % (save_dir, sig, epoch))
    torch.save(all_input_with_signal, '%s/result_5/%s/all_input_with_signal_%d.pickle' % (save_dir, sig, epoch))
    torch.save(all_signal_rec, '%s/result_5/%s/all_signal_rec_%d.pickle' % (save_dir, sig, epoch))
    torch.save(all_gen_img_with_signal, '%s/result_5/%s/all_gen_img_with_signal_%d.pickle' % (save_dir, sig, epoch))
    torch.save(all_signal_fake, '%s/result_5/%s/all_signal_fake_%d.pickle' % (save_dir, sig, epoch))
    
    
