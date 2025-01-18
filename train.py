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
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau, CosineAnnealingLR
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

import wandb

wandb.init(
    project="expression_classifier",  # 替換為你的專案名稱
    name="with ID classifier",     # 替換為你的實驗名稱（可選）
    
)
class Args(dict):
        __setattr__ = dict.__setitem__
        __getattr__ = dict.__getitem__

args = {
    'device' : 'cuda',
    'batch_size' : int(8),
    'epoch' : 10000,
    'neutral_path' : r".\data\neutral_crop_align_128",
    'smile_path' : r".\data\smile_crop_align_128",
    'neutral_test_path' : r".\data\test_data\test_neutral128",
    'smile_test_path' : r".\data\test_data\test_smile128",   
    'expression_path' : r".\model\expression_classifier\attention0.5",
    'Id_classifier_path' : r".\model\with_ID\Id_classifier_model",
    'neutral_landmark_path' : r".\data\neutral_feature_points",
    'smile_landmark_path' : r".\data\smile_feature_points",
}
args = Args(args)

model_Enc = EASNNetwork().to("cuda")
model_Id = IdClassifier().to("cuda")
train_dataset = Ds(args.neutral_path, args.smile_path)
test_dataset = Ds(args.neutral_test_path, args.smile_test_path)
# dataset = Ds(args.neutral_path, args.smile_path)
# train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=38)



ID_loss = IdClassifyLoss(args)
loss = Expression_Loss(args)
opt_Enc = torch.optim.Adam(model_Enc.parameters(), lr=0.001, weight_decay=0.01)
opt_Id =  torch.optim.Adam(model_Id.parameters(), lr=0.001)

scheduler_Enc = ReduceLROnPlateau(opt_Enc, mode='min', factor=0.8, patience=8, min_lr=1e-6, cooldown=3, threshold=1e-4)
#opt_Id = torch.optim.RAdam(model_Id.parameters(), lr=0.001)
scheduler_Id = ReduceLROnPlateau(opt_Id, mode='min', factor=0.5, patience=5, min_lr=1e-6)
#params = list(model_Enc.parameters()) + list(model_Id.parameters())
#opt = torch.optim.RAdam(params,lr=0.001)
labels_neutral = torch.zeros(args.batch_size, dtype=torch.long, device=args.device)
labels_smile = torch.ones(args.batch_size, dtype=torch.long, device=args.device)
attention_argument = float(0.5)
Id_argument = float(1.0)
kfold = KFold(n_splits = 4, shuffle = True)
for train_index, valid_index in kfold.split(train_dataset):
    train_subset = Subset(train_dataset, train_index)
    valid_subset = Subset(train_dataset, valid_index)

        # 創建新的 DataLoader 來處理這些子集
    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_subset, batch_size=args.batch_size, shuffle=False, drop_last=True)
    for epoch in range(args.epoch):
        
        model_Enc.train()
        model_Id.train()
        
        with tqdm(train_loader, unit="batch", desc=f"Epoch {epoch + 1}") as tepoch:
            for batch in tepoch:
                tepoch.set_description(f"Epoch {epoch}")
                
                input_neutral = batch['neutral']  
                #print("input neutral", input_neutral.shape)      
                input_smile = batch['smile']
                
                output_neutral, x_encoded = model_Enc(input_neutral)
                output_smile, y_encoded = model_Enc(input_smile)                
                output_Id = model_Id(x_encoded, y_encoded)                
                # loss_express = loss_expfn(output_neutral, 'neutral') + loss_expfn(output_smile, 'smile')
                x_encoded_detached = x_encoded.detach()  # 創建新的張量，切斷梯度連接
                y_encoded_detached = y_encoded.detach()               
                model_Id.eval()        
                model_Enc.eval()             
                com_neutral, com_smile = Grad_CAM(model_Enc, batch)     
                model_Id.train()
                model_Enc.train()
            
                # loss_attention_neutral = loss_attention(complentary_neutral_output, 'neutral')
                # loss_attention_smile = loss_attention(complentart_smile_output, 'smile')
                #total_loss = loss_express + 0.5*loss_Id + loss_attention_neutral + loss_attention_smile
                
                expression_total_loss = loss(output_neutral, output_smile, com_neutral, com_smile, attention_argument)
                ID_classifierLoss = ID_loss(output_Id, batch['same_id'])
                total_loss = ID_classifierLoss + expression_total_loss
                wandb.log({"expression_loss": expression_total_loss.item(), "epoch": epoch})
            
                opt_Enc.zero_grad()
                total_loss.backward()
                #expression_total_loss.backward()
                # expression_total_loss.backward(retain_graph=True)  # 保留计算图
                # ID_classifierLoss.backward()
                opt_Enc.step()
                # model_Id.train()
                # output_Id = model_Id(x_encoded_detached, y_encoded_detached)
                # ID_classifierLoss = ID_loss(output_Id, batch['same_id'])
                # wandb.log({"Id_classifierloss": ID_classifierLoss.item(), "epoch": epoch})
                # opt_Id.zero_grad()
                # ID_classifierLoss.backward()
                # opt_Id.step()
                
                _, predicted_neutral = torch.max(output_neutral, 1)
                _, predicted_smile = torch.max(output_smile, 1)
                    
                predicted_all = torch.cat((predicted_neutral, predicted_smile), dim=0)

                    # 将两个真实标签合并
                labels_all = torch.cat((labels_neutral, labels_smile), dim=0)

                    # 计算 F1 Score
                f1 = f1_score(labels_all.cpu().numpy(), predicted_all.cpu().numpy(), average='binary')
                wandb.log({"f1_score": f1.item(), "epoch": epoch})
                
                
                tepoch.set_postfix(loss=expression_total_loss.item())
                
                model_Enc.eval()
                #model_Id.eval()
            opt_Enc.zero_grad()
            model_Enc.eval()
            with torch.no_grad():
                valid_total_loss = 0
                totalID_loss = 0
                labels_neutral = torch.zeros(args.batch_size, dtype=torch.long, device=args.device)
                labels_smile = torch.ones(args.batch_size, dtype=torch.long, device=args.device)
                for batch in valid_loader:
                    input_neutral = batch['neutral']
                    input_smile = batch['smile']
                    output_neutral, x_encoded = model_Enc(input_neutral)
                    output_smile, y_encoded = model_Enc(input_smile)
                    with torch.enable_grad():  # 启用梯度计算
                        com_neutral, com_smile = Grad_CAM(model_Enc, batch)
                    
                    _, predicted_neutral = torch.max(output_neutral, 1)
                    _, predicted_smile = torch.max(output_smile, 1)
                    
                    predicted_all = torch.cat((predicted_neutral, predicted_smile), dim=0)

                    # 将两个真实标签合并
                    labels_all = torch.cat((labels_neutral, labels_smile), dim=0)

                    # 计算 F1 Score
                    f1 = f1_score(labels_all.cpu().numpy(), predicted_all.cpu().numpy(), average='binary')

                    print(f"valid expression f1score {f1}")
                    wandb.log({"validation_f1_score": f1.item(), "epoch": epoch})
                    output_Id = model_Id(x_encoded, y_encoded) 
                    probs = torch.sigmoid(output_Id) 
                
                    # predictions = (probs > 0.5).float()
                    # same_id_f1 = f1_score(batch['same_id'].cpu().numpy(), predictions.cpu().numpy())
                    # print(f"valid ID f1score {same_id_f1}")
                    valid_loss_express = loss(output_neutral, output_smile, com_neutral, com_smile, attention_argument)                     
                    
                    wandb.log({"validation loss express": valid_loss_express.item(), "epoch": epoch})
                    valid_total_loss += valid_loss_express
                    
                    # ID_classifierLoss = ID_loss(output_Id, batch['same_id'])
                    
                    #totalID_loss += ID_classifierLoss
                    # complemtary_neutral, complemtary_smile = Grad_CAM(model_Enc, batch)      
                    # total_loss = loss(output_neutral, output_smile, output_Id, batch, complemtary_neutral, complemtary_smile)         
                
                valid_average_loss = valid_total_loss/len(valid_loader)           
                scheduler_Enc.step(valid_average_loss)
                lr = opt_Enc.param_groups[0]['lr']
                wandb.log({"lr": lr, "epoch": epoch})
                #totalID_loss = totalID_loss/(len(valid_loader))
                
                #scheduler_Id.step(totalID_loss)
                test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
                for batch in test_loader:
                    input_neutral = batch['neutral']
                    input_smile = batch['smile']
                    
                    output_smile, x_encoded = model_Enc(input_smile)
                    output_neutral, y_encoded = model_Enc(input_neutral)
                    with torch.enable_grad():
                        com_neutral, com_smile = Grad_CAM(model_Enc, batch) 
                    _, predicted_smile = torch.max(output_smile, 1)
                    _, predicted_neutral = torch.max(output_neutral, 1)
                    
                    
                    predicted_all = torch.cat((predicted_smile, predicted_neutral), dim=0)

                    # 将两个真实标签合并
                    labels_all = torch.cat((labels_smile, labels_neutral), dim=0)

                    # 计算 F1 Score
                    f1 = f1_score(labels_all.cpu().numpy(), predicted_all.cpu().numpy(), average='binary')
                    # probabilities = torch.sigmoid(output_Id)
                    #output_Id = model_Id(x_encoded, y_encoded) 
                    probs = torch.sigmoid(output_Id) 
                
                    predictions = (probs > 0.5).float()
                    test_loss_express = loss(output_neutral, output_smile, com_neutral, com_smile, attention_argument)
                    wandb.log({"test loss express": test_loss_express.item(), "epoch": epoch})
                    # # 設定閾值為 0.5，轉為二進制標籤
                    # predicted_labels = (probabilities > 0.5).int()
                    same_id_f1 = f1_score(batch['same_id'].cpu().numpy(), predictions.cpu().numpy())
                    print(f"test expression f1score {f1}")
                    #print(f"test ID f1score {same_id_f1}")
                    #print(f"same Id {same_id_f1}")
        
        
            #complemtary_neutral, complemtart_smile = Grad_CAM(model_Enc, batch)
        if epoch % 500 == 0:
            torch.save(model_Enc.state_dict(), f"{args.expression_path}_epoch{epoch}.pth")
            #torch.save(model_Id.state_dict(), f"{args.Id_classifier_path}_epoch{epoch}.pth")
            print(f"successful save model")

 
         
