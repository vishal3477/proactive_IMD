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
import encoder2_1
from model import *
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from sklearn import metrics

import argparse
from functools import partial
import json
import traceback


import imlib as im
import numpy as np
import pylib
import tensorflow as tf
import tflib as tl

#import data
from STGAN import models

import os

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





opt = parser.parse_args()
print(opt)
print("Random Seed: ", opt.seed)

size=opt.image_size
set_size=opt.set_size
b_s=opt.batch_size
m=opt.template_strength

with open('./STGAN/output/%s/setting.txt' % 128) as f:
    args = json.load(f)

# model
atts = args['atts']
n_att = len(atts)
img_size = args['img_size']
shortcut_layers = args['shortcut_layers']
inject_layers = args['inject_layers']
enc_dim = args['enc_dim']
dec_dim = args['dec_dim']
dis_dim = args['dis_dim']
dis_fc_dim = args['dis_fc_dim']
enc_layers = args['enc_layers']
dec_layers = args['dec_layers']
dis_layers = args['dis_layers']

label = args['label']
use_stu = args['use_stu']
stu_dim = args['stu_dim']
stu_layers = args['stu_layers']
stu_inject_layers = args['stu_inject_layers']
stu_kernel_size = args['stu_kernel_size']
stu_norm = args['stu_norm']
stu_state = args['stu_state']
multi_inputs = args['multi_inputs']
rec_loss_weight = args['rec_loss_weight']
one_more_conv = args['one_more_conv']

img = None
print('Using selected images:', img)

gpu = 'all'
if gpu != 'all':
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu

#### testing
# multiple attributes
test_atts = None
test_ints = None
if test_atts is not None and test_ints is None:
    test_ints = [1 for i in range(len(test_atts))]
# single attribute
test_int = 1.0
# slide attribute
test_slide = False
n_slide = 10
test_att = None
test_int_min = -1.0
test_int_max = 1.0

thres_int = args['thres_int']
# others
use_cropped_img = args['use_cropped_img']
experiment_name = 128


device=torch.device("cuda:0")
torch.backends.deterministic = True
torch.manual_seed(opt.seed)
torch.cuda.manual_seed_all(opt.seed)

sig = str(datetime.datetime.now())



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
    
    
train_path=opt.data_train
test_path=opt.data_test
save_dir=opt.savedir
os.makedirs('%s/logs/%s' % (save_dir, sig), exist_ok=True)
os.makedirs('%s/result_5/%s' % (save_dir, sig), exist_ok=True)

best_acc = 0
start_epoch = 1

transform_train = transforms.Compose([
transforms.Resize((size,size)),
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
optimizer_2 = torch.optim.Adam(encoder2_model.parameters(), lr=0.00001)



sess = tl.session()

# models
Genc = partial(models.Genc, dim=enc_dim, n_layers=enc_layers, multi_inputs=multi_inputs)
Gdec = partial(models.Gdec, dim=dec_dim, n_layers=dec_layers, shortcut_layers=shortcut_layers,
               inject_layers=inject_layers, one_more_conv=one_more_conv)
Gstu = partial(models.Gstu, dim=stu_dim, n_layers=stu_layers, inject_layers=stu_inject_layers,
               kernel_size=stu_kernel_size, norm=stu_norm, pass_state=stu_state)

# inputs
xa_sample = tf.placeholder(tf.float32, shape=[None, img_size, img_size, 3])
_b_sample = tf.placeholder(tf.float32, shape=[None, n_att])
raw_b_sample = tf.placeholder(tf.float32, shape=[None, n_att])

# sample
test_label = _b_sample - raw_b_sample if label == 'diff' else _b_sample
if use_stu:
    x_sample = Gdec(Gstu(Genc(xa_sample, is_training=False),
                         test_label, is_training=False), test_label, is_training=False)
else:
    x_sample = Gdec(Genc(xa_sample, is_training=False), test_label, is_training=False)

# ==============================================================================
# =                                    test                                    =
# ==============================================================================

# initialization
ckpt_dir = './STGAN/output/%s/checkpoints' % experiment_name
tl.load_checkpoint(ckpt_dir, sess)

signal_set_init=vector_var(set_size, size).cuda()
optimizer_3 = torch.optim.Adam(signal_set_init.parameters(), lr=0.00001)

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

'''
state1 = torch.load('/mnt/gs18/scratch/users/asnanivi/runs/result_5/2021-06-02 23:27:33.963942/19_model.pickle')
encoder2_model.load_state_dict(state1['state_dict_encoder2'])
optimizer_2.load_state_dict(state1['optimizer_2'])
signal_set_init.load_state_dict(state1['state_dict_signal_set'])
optimizer_3.load_state_dict(state1['optimizer_3'])
'''


def train(batch):
    
    encoder2_model.train()
    #gan.train()
    #prob =encoder1_model(batch.type(torch.cuda.FloatTensor))
    signal_sel=torch.randint(0,set_size,(b_s,))
    print(signal_sel)
    
    signal_est=signal_set[signal_sel.type(torch.cuda.LongTensor),:]
 
    signal_est_channel_rem=signal_est[:,0,:].clone()
    signal_est_fs=torch.rfft(signal_est_channel_rem, signal_ndim=2, onesided=False)

    signal_est_fs[:,:,:,0],signal_est_fs[:,:,:,1]=fftshift(signal_est_fs[:,:,:,0],signal_est_fs[:,:,:,1])
    signal_est_fs_shift=torch.sqrt(signal_est_fs[:,:,:,0]**2+signal_est_fs[:,:,:,1]**2)
    
    n=50
    (_,w,h)=signal_est_fs_shift.shape
    half_w, half_h = int(w/2), int(h/2)

    signal_est_fs_low_freq=signal_est_fs_shift[:,half_w-n:half_w+n+1,half_h-n:half_h+n+1].clone()
    target_zero = torch.zeros(signal_est_fs_low_freq.shape, dtype=torch.float32).to(device) 
   
    signal_est_red=m*signal_est.clone() 
    signal_set_red=m*signal_set.clone() 
    input_with_signal=batch.type(torch.cuda.FloatTensor) + signal_est_red
    
    batch_denorm=(input_with_signal*0.1269)+0.6490
    signal_rec=encoder2_model(input_with_signal)

    b_sample=[[0,1,0,1,0,0,0,0,0,0,0,0,0],[0,1,0,1,0,0,0,0,0,0,0,0,0],[0,1,0,1,0,0,0,0,0,0,0,0,0],[0,1,0,1,0,0,0,0,0,0,0,0,0]]
    raw_b_sample_1=[[0,1,0,0,0,0,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0,0,0,0,0,0]]
    gen_img_with_signal = torch.tensor(sess.run(x_sample, feed_dict={xa_sample: batch_denorm.cpu().detach().permute(0,2,3,1),_b_sample: b_sample,raw_b_sample: raw_b_sample_1}))
    
    signal_fake=encoder2_model(gen_img_with_signal.permute(0,3,1,2).type(torch.cuda.FloatTensor) )
    
    
    
    
    
    n=25
    zero=torch.zeros([batch.shape[0],1,batch.shape[2],batch.shape[3]], dtype=torch.float32).to(device) 
    loss1= 200*l2(signal_est,zero)
    loss2=0
    signal_set_norm=signal_set.clone()
    for i in range(set_size):
        signal_set_norm[i]=(signal_set[i]-torch.min(signal_set[i]))/(torch.max(signal_set[i])-torch.min(signal_set[i]))
    signal_set_norm[torch.isnan(signal_set_norm)]=0
    signal_set_norm_red=m*signal_set_norm.clone()
    for i in range(set_size-1):
        for j in range(i+1):
            
            loss2+=cos1(signal_set_norm[i+1,:].reshape( -1), signal_set_norm[j,:].reshape( -1))
            #print(loss2)
       
    loss2_tot=30* loss2
    print(signal_set.shape)
    
          
    loss3=(1. - cos(signal_est_red.reshape( signal_est_red.size(0), -1), signal_rec.reshape( signal_rec.size(0), -1)))
    loss3_tot=5*torch.sum(loss3)
    #loss3=l2(signal_est, signal_rec)
    #loss3_tot=100*loss3
    loss4=0.003*l2(signal_est_fs_low_freq, target_zero)
    loss5=0
    signal_fake_norm=signal_fake.clone()
    for i in range(b_s):
        signal_fake_norm[i]=(signal_fake[i]-torch.min(signal_fake[i]))/(torch.max(signal_fake[i])-torch.min(signal_fake[i]))
    signal_fake_norm[torch.isnan(signal_fake_norm)]=0
    
    for i in range(set_size):
        loss5+=cos(signal_set_norm_red[i,:].reshape( -1).unsqueeze(0), signal_fake_norm.reshape(signal_fake.size(0), -1))
    loss5_tot=10*torch.max(loss5)
    #loss6=50* l2(signal_fake,zero)
    loss=loss1+loss2_tot+loss3_tot+loss4+loss5_tot
    print(loss, loss1, loss2_tot, loss3_tot, loss4, loss5_tot)
    

    
    optimizer_2.zero_grad()
    optimizer_3.zero_grad()
    
    loss.backward()
    
    
    optimizer_2.step()
    optimizer_3.step()
    
    signal_fake_flattened=signal_fake.reshape(signal_fake.size(0), -1)
    signal_rec_flattened=signal_rec.reshape(signal_rec.size(0), -1)
    signal_set_flattened=signal_set_red.reshape(signal_set.size(0), -1)
    
    dist_fake=torch.zeros([batch.shape[0],set_size,2], dtype=torch.float32).to(device)
    for i in range(set_size):
        dist_fake[:,i,0]=cos(signal_set_red[i,:].reshape( -1).unsqueeze(0), signal_fake.reshape(signal_fake.size(0), -1))
        dist_fake[:,i,1]=cos(signal_set_red[i,:].reshape( -1).unsqueeze(0), signal_rec.reshape(signal_fake.size(0), -1))
    
    
    return  signal_set,batch.type(torch.cuda.FloatTensor) ,input_with_signal, signal_rec,gen_img_with_signal,signal_fake, dist_fake, loss.item(), loss1.item(),loss2_tot.item(),loss3_tot.item(),loss4.item(),loss5_tot.item()


def test(batch):
   
    
    
    with torch.no_grad():
        
        signal_sel=torch.randint(0,5,(b_s,))

        signal_est=signal_set[signal_sel.type(torch.cuda.LongTensor),:]


        
        signal_est_red=m*signal_est.clone() 
        signal_set_red=m*signal_set.clone()
        input_with_signal=batch.type(torch.cuda.FloatTensor) + signal_est_red

        signal_rec=encoder2_model(input_with_signal)

        batch_denorm=(input_with_signal*0.1269)+0.6490
        signal_rec=encoder2_model(input_with_signal)

        b_sample=[[0,1,0,1,0,0,0,0,0,0,0,0,0],[0,1,0,1,0,0,0,0,0,0,0,0,0],[0,1,0,1,0,0,0,0,0,0,0,0,0],[0,1,0,1,0,0,0,0,0,0,0,0,0]]
        raw_b_sample_1=[[0,1,0,0,0,0,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0,0,0,0,0,0]]
        gen_img_with_signal = torch.tensor(sess.run(x_sample, feed_dict={xa_sample: batch_denorm.cpu().detach().permute(0,2,3,1),_b_sample: b_sample,raw_b_sample: raw_b_sample_1}))

        signal_fake=encoder2_model(gen_img_with_signal.permute(0,3,1,2).type(torch.cuda.FloatTensor) )

        
        dist_fake=torch.zeros([batch.shape[0],set_size,2], dtype=torch.float32).to(device)
        for i in range(set_size):
            dist_fake[:,i,0]=cos(signal_set_red[i,:].reshape( -1).unsqueeze(0), signal_fake.reshape(signal_fake.size(0), -1))
            dist_fake[:,i,1]=cos(signal_set_red[i,:].reshape( -1).unsqueeze(0), signal_rec.reshape(signal_fake.size(0), -1))
    
    return  signal_set,batch.type(torch.cuda.FloatTensor) ,input_with_signal, signal_rec,gen_img_with_signal.permute(0,3,1,2).type(torch.cuda.FloatTensor),signal_fake, dist_fake


epochs=40

for epoch in range(epochs):
    all_y=[]
    all_y_test=[]
    flag=0
    flag1=0
    count=0
    itr=0
    for batch_idx, (inputs,labels) in enumerate(train_loader):
        
        signal_set,input_images ,input_with_signal, signal_rec,gen_img_with_signal,signal_fake, dist_fake=train(Variable(torch.FloatTensor(inputs)))
        
      
        count+=b_s
        
        if flag==0:
            all_dist_fake=dist_fake.detach()
            
            all_signal_set=signal_set.detach()
            all_input_images=input_images.detach()
            all_input_with_signal=input_with_signal.detach()
            all_signal_rec=signal_rec.detach()
            all_gen_img_with_signal=gen_img_with_signal.detach()
            all_signal_fake=signal_fake.detach()
            flag=1
        else:
            all_dist_fake=torch.cat([all_dist_fake,dist_fake.detach()], dim=0)
            if count>161500:
                
                all_signal_set=torch.cat([all_signal_set,signal_set.detach()], dim=0)
                all_input_images=torch.cat([all_input_images,input_images.detach()], dim=0)
                all_input_with_signal=torch.cat([all_input_with_signal,input_with_signal.detach()], dim=0)
                all_signal_rec=torch.cat([all_signal_rec,signal_rec.detach()], dim=0)
                all_gen_img_with_signal=torch.cat([all_gen_img_with_signal,gen_img_with_signal.detach()], dim=0)
                all_signal_fake=torch.cat([all_signal_fake,signal_fake.detach()], dim=0)
    print('epoch=',epoch)
   
    
   
    torch.save(all_dist_fake, '%s/result_5/%s/all_dist_fake_%d.pickle' % (save_dir, sig, epoch))
    
    torch.save(all_signal_set, '%s/result_5/%s/all_signal_set_%d.pickle' % (save_dir, sig, epoch))
    torch.save(all_input_images, '%s/result_5/%s/all_input_images_%d.pickle' % (save_dir, sig, epoch))
    torch.save(all_input_with_signal, '%s/result_5/%s/all_input_with_signal_%d.pickle' % (save_dir, sig, epoch))
    torch.save(all_signal_rec, '%s/result_5/%s/all_signal_rec_%d.pickle' % (save_dir, sig, epoch))
    torch.save(all_gen_img_with_signal, '%s/result_5/%s/all_gen_img_with_signal_%d.pickle' % (save_dir, sig, epoch))
    torch.save(all_signal_fake, '%s/result_5/%s/all_signal_fake_%d.pickle' % (save_dir, sig, epoch))
    torch.save(state, '%s/result_5/%s/%d_model.pickle' % (save_dir, sig, epoch))
    print("Save Model: {:d}".format(epoch))
    

    for batch_idx_test, (inputs_test,labels_test) in enumerate(test_loader):
    
        signal_set,input_images ,input_with_signal, signal_rec,gen_img_with_signal,signal_fake, dist_fake=test(Variable(torch.FloatTensor(inputs_test)))
        
        
        if flag1==0:
            all_dist_test=dist_fake.detach()
            
            all_signal_set_test=signal_set.detach()
            all_input_images_test=input_images.detach()
            all_input_with_signal_test=input_with_signal.detach()
            all_signal_rec_test=signal_rec.detach()
            all_gen_img_with_signal_test=gen_img_with_signal.detach()
            all_signal_fake_test=signal_fake.detach()
            flag1=1
        else:
            all_dist_test=torch.cat([all_dist_test,dist_fake.detach()], dim=0)
            
            
    
    torch.save(all_dist_test, '%s/result_5/%s/dist_test_%d.pickle' % (save_dir, sig, epoch))     
   
    torch.save(all_signal_set_test, '%s/result_5/%s/all_signal_set_test_%d.pickle' % (save_dir, sig, epoch))
    torch.save(all_input_images_test, '%s/result_5/%s/all_input_images_test_%d.pickle' % (save_dir, sig, epoch))
    torch.save(all_input_with_signal_test, '%s/result_5/%s/all_input_with_signal_test_%d.pickle' % (save_dir, sig, epoch))
    torch.save(all_signal_rec_test, '%s/result_5/%s/all_signal_rec_test_%d.pickle' % (save_dir, sig, epoch))
    torch.save(all_gen_img_with_signal_test, '%s/result_5/%s/all_gen_img_with_signal_test_%d.pickle' % (save_dir, sig, epoch))
    torch.save(all_signal_fake_test, '%s/result_5/%s/all_signal_fake_test_%d.pickle' % (save_dir, sig, epoch))
    