from Enc import EASNNetwork, IdClassifier
from sklearn.model_selection import train_test_split
from Dataset import Dataset as Ds
import torch
from torch.utils.data import random_split, DataLoader
from loss import ExpressionClassifyLoss, IdClassifyLoss, AttentionLoss, TotalLoss
from Grad_CAM import Grad_CAM
from torch.optim import Adam
from tqdm import tqdm
import time
import numpy as np
class Args(dict):
        __setattr__ = dict.__setitem__
        __getattr__ = dict.__getitem__

args = {
    'device' : 'cuda',
    'batch_size' : int(2)
}
args = Args(args)

neutral_path = r"E:\style_exprGAN\data\neutral_crop"
smile_path = r"E:\style_exprGAN\data\smile_crop"
neutral_test_path = r"E:\style_exprGAN\data\neutral_crop_test"
smile_test_path = r"E:\style_exprGAN\data\smile_crop_test"
export_path = r"E:\style_exprGAN\model\\model_weights.pth"
neutral_landmark_path = r"E:\style_exprGAN\data\neutral_feature_points"
smile_landmark_path =  r"E:\style_exprGAN\data\smile_feature_points"


model_Enc = EASNNetwork().to("cuda")
model_Id = IdClassifier(model_Enc.get_flatten_size()).to("cuda")
dataset = Ds(neutral_path, smile_path)
#test_test_dataset = Ds(neutral_test_path, smile_test_path)

#Set a seed for reproducibility
#torch.manual_seed(35)

train_loader, test_loader= train_test_split(dataset, test_size=0.4, random_state=42)
valid_loader, test_loader= train_test_split(test_loader, test_size=0.5, random_state=42)


#Define the split ratio
# train_ratio = 0.6
# train_size = int(len(dataset) * train_ratio)
# test_size = len(dataset) - train_size


# train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
# train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
# #test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)
# test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

loss_expfn = ExpressionClassifyLoss(args)
loss_Idfn = IdClassifyLoss(args)
loss_attention = AttentionLoss(args)
loss = TotalLoss(args)
# opt_Enc = Adam(model_Enc.parameters(), lr=0.002)
# opt_Id = Adam(model_Id.parameters(), lr=0.0002)
params = list(model_Enc.parameters()) + list(model_Id.parameters())
opt = torch.optim.RAdam(params,lr=0.001)

for epoch in range(1000):
    
    model_Enc.train()
    model_Id.train()
    with tqdm(train_loader, unit="batch", desc=f"Epoch {epoch + 1}") as tepoch:
        for batch in tepoch:
            tepoch.set_description(f"Epoch {epoch}")
            
            input_neutral = batch['neutral']        
            input_smile = batch['smile']
            output_neutral, x_encoded = model_Enc(input_neutral)
            output_smile, y_encoded = model_Enc(input_smile)
            
            output_Id = model_Id(x_encoded, y_encoded)
            
            # loss_express = loss_expfn(output_neutral, 'neutral') + loss_expfn(output_smile, 'smile')
            # loss_Id = loss_Idfn(output_Id, batch['same_id'])                 
                    
            model_Enc.eval()
            model_Id.eval()  
            com_neutral, com_smile = Grad_CAM(model_Enc, batch)        
            
            model_Enc.train()
            model_Id.train()
            # loss_attention_neutral = loss_attention(complentary_neutral_output, 'neutral')
            # loss_attention_smile = loss_attention(complentart_smile_output, 'smile')
            #total_loss = loss_express + 0.5*loss_Id + loss_attention_neutral + loss_attention_smile
            total_loss = loss(output_neutral, output_smile, output_Id, batch, com_neutral, com_smile)
            opt.zero_grad()
            total_loss.backward()
            opt.step()  
            print(total_loss.item())
            tepoch.set_postfix(loss=total_loss.item())
            
        model_Enc.eval()
        model_Id.eval()
        with torch.no_grad():
             for batch in valid_loader:
                input_neutral = batch['neutral']
                input_smile = batch['smile']
                output_neutral, x_encoded = model_Enc(input_neutral)
                output_smile, y_encoded = model_Enc(input_smile)
                output_Id = model_Id(x_encoded, y_encoded)
                # input_combined = torch.cat((input_neutral, input_smile), dim=0)  # 合并两个输入
                # output_combined, encoded_combined = model_Enc(input_combined)
                
                # output_neutral = output_combined[:2]  # 根据批次大小分割
                # output_smile = output_combined[2:]
                # x_encoded = encoded_combined[:2]
                # y_encoded = encoded_combined[2:]
                # output_Id = model_Id(x_encoded, y_encoded)
                
                loss_express = loss_expfn(output_neutral, 'neutral') + loss_expfn(output_smile, 'smile')
                loss_Id = loss_Idfn(output_Id, batch['same_id'])         
                    
                total_loss = loss_express + loss_Id
                # complemtary_neutral, complemtary_smile = Grad_CAM(model_Enc, batch)      
                # total_loss = loss(output_neutral, output_smile, output_Id, batch, complemtary_neutral, complemtary_smile)         
        
                print(f"test loss:{total_loss}")
        #complemtary_neutral, complemtart_smile = Grad_CAM(model_Enc, batch)
       
torch.save(model_Enc.state_dict(), export_path)
model_Enc.eval()
data_iter = iter(train_loader)   # 获取 DataLoader 的迭代器
first_batch = next(data_iter)    # 获取第一个批次数据

 
         
