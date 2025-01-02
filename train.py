from Enc import EASNNetwork, IdClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from Dataset import Dataset as Ds
import torch
from torch.utils.data import random_split, DataLoader, Subset
from loss import ExpressionClassifyLoss, IdClassifyLoss, AttentionLoss, Expression_Loss
from Grad_CAM import Grad_CAM
from torch.optim import Adam
from tqdm import tqdm
import time
import numpy as np
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau
from sklearn.metrics import f1_score

class Args(dict):
        __setattr__ = dict.__setitem__
        __getattr__ = dict.__getitem__

args = {
    'device' : 'cuda',
    'batch_size' : int(4),
    'epoch' : 10000,
    'neutral_path' : r".\data\neutral_crop_128",
    'smile_path' : r".\data\smile_crop_128",
    'neutral_test_path' : r".\data\test_data\test_neutral",
    'smile_test_path' : r".\data\test_data\test_smile",
    'export_path' : r".\model\with_ID\model_weights",
    'neutral_landmark_path' : r".\data\neutral_feature_points",
    'smile_landmark_path' : r".\data\smile_feature_points",
}
args = Args(args)

model_Enc = EASNNetwork().to("cuda")
model_Id = IdClassifier().to("cuda")
train_dataset = Ds(args.neutral_path, args.smile_path)
test_dataset = Ds(args.neutral_test_path, args.smile_test_path)

kfold = KFold(n_splits = 4, shuffle = True)

ID_loss = IdClassifyLoss(args)
loss = Expression_Loss(args)
opt_Enc = torch.optim.Adam(model_Enc.parameters(), lr=0.001)
opt_Id =  torch.optim.Adam(model_Id.parameters(), lr=0.001)
scheduler_Enc = ReduceLROnPlateau(opt_Enc, mode='min', factor=0.5, patience=5, min_lr=1e-6)
#opt_Id = torch.optim.RAdam(model_Id.parameters(), lr=0.001)
scheduler_Id = ReduceLROnPlateau(opt_Id, mode='min', factor=0.5, patience=5, min_lr=1e-6)
#params = list(model_Enc.parameters()) + list(model_Id.parameters())
#opt = torch.optim.RAdam(params,lr=0.001)
attention_argument = float(0.5)
Id_argument = float(0.0)
for epoch in range(args.epoch):
    labels_neutral = torch.zeros(args.batch_size, dtype=torch.long, device=args.device)
    labels_smile = torch.ones(args.batch_size, dtype=torch.long, device=args.device)
    model_Enc.train()
    model_Id.train()
    for train_index, valid_index in kfold.split(train_dataset):
        train_subset = Subset(train_dataset, train_index)
        valid_subset = Subset(train_dataset, valid_index)

        # 創建新的 DataLoader 來處理這些子集
        train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, drop_last=True)
        valid_loader = DataLoader(valid_subset, batch_size=args.batch_size, shuffle=False, drop_last=True)
    with tqdm(train_loader, unit="batch", desc=f"Epoch {epoch + 1}") as tepoch:
        for batch in tepoch:
            tepoch.set_description(f"Epoch {epoch}")
            
            input_neutral = batch['neutral']        
            input_smile = batch['smile']
            output_neutral, x_encoded = model_Enc(input_neutral)
            output_smile, y_encoded = model_Enc(input_smile)                
            output_Id = model_Id(x_encoded, y_encoded)                
            # loss_express = loss_expfn(output_neutral, 'neutral') + loss_expfn(output_smile, 'smile')
                           
                    
            model_Enc.eval()
            model_Id.eval()              
            com_neutral, com_smile = Grad_CAM(model_Enc, batch)     
            
            model_Enc.train()
            model_Id.train()
            # loss_attention_neutral = loss_attention(complentary_neutral_output, 'neutral')
            # loss_attention_smile = loss_attention(complentart_smile_output, 'smile')
            #total_loss = loss_express + 0.5*loss_Id + loss_attention_neutral + loss_attention_smile
            ID_classifierLoss = ID_loss(output_Id, batch['same_id'])
            expression_total_loss = loss(output_neutral, output_smile, com_neutral, com_smile, attention_argument)
            total_loss = ID_classifierLoss + expression_total_loss
            opt_Enc.zero_grad()
            opt_Id.zero_grad()
            expression_total_loss.backward(retain_graph=True)  # 保留计算图
            ID_classifierLoss.backward()
            opt_Enc.step()
            opt_Id.step()
            _, predicted_neutral = torch.max(output_neutral, 1)
            _, predicted_smile = torch.max(output_smile, 1)
                
            predicted_all = torch.cat((predicted_neutral, predicted_smile), dim=0)

                # 将两个真实标签合并
            labels_all = torch.cat((labels_neutral, labels_smile), dim=0)

                # 计算 F1 Score
            f1 = f1_score(labels_all.cpu().numpy(), predicted_all.cpu().numpy(), average='binary')
            
            print(f"expression f1score {f1}\n")
            
            tepoch.set_postfix(loss=expression_total_loss.item())
            
            model_Enc.eval()
            model_Id.eval()
        
        with torch.no_grad():
            total_loss = 0
            totalID_loss = 0
            labels_neutral = torch.zeros(args.batch_size, dtype=torch.long, device=args.device)
            labels_smile = torch.ones(args.batch_size, dtype=torch.long, device=args.device)
            for batch in valid_loader:
                input_neutral = batch['neutral']
                input_smile = batch['smile']
                output_neutral, x_encoded = model_Enc(input_neutral)
                output_smile, y_encoded = model_Enc(input_smile)
                
                _, predicted_neutral = torch.max(output_neutral, 1)
                _, predicted_smile = torch.max(output_smile, 1)
                
                predicted_all = torch.cat((predicted_neutral, predicted_smile), dim=0)

                # 将两个真实标签合并
                labels_all = torch.cat((labels_neutral, labels_smile), dim=0)

                # 计算 F1 Score
                f1 = f1_score(labels_all.cpu().numpy(), predicted_all.cpu().numpy(), average='binary')

                print(f"valid expression f1score {f1}")
                
                output_Id = model_Id(x_encoded, y_encoded) 
                probs = torch.sigmoid(output_Id) 
               
                predictions = (probs > 0.5).float()
                same_id_f1 = f1_score(batch['same_id'].cpu().numpy(), predictions.cpu().numpy())
                print(f"valid ID f1score {same_id_f1}")
                loss_express = loss(output_neutral, output_smile, com_neutral, com_smile, attention_argument)                     
                ID_classifierLoss = ID_loss(output_Id, batch['same_id'])
                total_loss += loss_express
                totalID_loss += ID_classifierLoss
                # complemtary_neutral, complemtary_smile = Grad_CAM(model_Enc, batch)      
                # total_loss = loss(output_neutral, output_smile, output_Id, batch, complemtary_neutral, complemtary_smile)         

                
            total_loss = total_loss/(len(valid_loader))
            totalID_loss = totalID_loss/(len(valid_loader))
            scheduler_Enc.step(total_loss)
            scheduler_Id.step(totalID_loss)
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
            for batch in test_loader:
                input_neutral = batch['neutral']
                input_smile = batch['smile']
                
                output_smile, x_encoded = model_Enc(input_smile)
                output_neutral, y_encoded = model_Enc(input_neutral)
                _, predicted_smile = torch.max(output_smile, 1)
                _, predicted_neutral = torch.max(output_neutral, 1)
                
                
                predicted_all = torch.cat((predicted_smile, predicted_neutral), dim=0)

                # 将两个真实标签合并
                labels_all = torch.cat((labels_smile, labels_neutral), dim=0)

                # 计算 F1 Score
                f1 = f1_score(labels_all.cpu().numpy(), predicted_all.cpu().numpy(), average='binary')
                # probabilities = torch.sigmoid(output_Id)
                output_Id = model_Id(x_encoded, y_encoded) 
                probs = torch.sigmoid(output_Id) 
               
                predictions = (probs > 0.5).float()
                # # 設定閾值為 0.5，轉為二進制標籤
                # predicted_labels = (probabilities > 0.5).int()
                same_id_f1 = f1_score(batch['same_id'].cpu().numpy(), predictions.cpu().numpy())
                print(f"test expression f1score {f1}")
                print(f"test ID f1score {same_id_f1}")
                #print(f"same Id {same_id_f1}")
                
        #complemtary_neutral, complemtart_smile = Grad_CAM(model_Enc, batch)
    if epoch % 500 == 0:
        torch.save(model_Enc.state_dict(), f"{args.export_path}_epoch{epoch}.pth")
        print(f"successful save model")

 
         
