from Enc import EASNNetwork, IdClassifier
from sklearn.model_selection import KFold
from Dataset import Dataset as Ds
import torch 
from torch.utils.data import DataLoader, Subset
from loss import IdClassifyLoss
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
    'epoch' : 5000,
    'neutral_path' : r".\data\neutral_crop",
    'smile_path' : r".\data\smile_crop",
    'neutral_test_path' : r".\data\test_data\test_neutral",
    'smile_test_path' : r".\data\test_data\test_smile",
    'export_path' : r".\model\ID_classifier\ID_classifier_weights",    
    'neutral_landmark_path' : r".\data\neutral_feature_points",
    'smile_landmark_path' : r".\data\smile_feature_points",
    
}
args = Args(args)

model_Enc = EASNNetwork()
model_Enc.load_state_dict(torch.load(r'.\model\test\model_weights_epoch4500.pth'))
model_Enc = model_Enc.to("cuda")
model_Id = IdClassifier().to("cuda")

train_dataset = Ds(args.neutral_path, args.smile_path)
test_dataset = Ds(args.neutral_test_path, args.smile_test_path)

kfold = KFold(n_splits = 4, shuffle = True)

loss = IdClassifyLoss(args)
opt_Id = torch.optim.Adam(model_Id.parameters(), lr=0.001)
scheduler = ReduceLROnPlateau(opt_Id, mode='min', factor=0.5, patience=5, min_lr=1e-6)
for epoch in range(5000):
    model_Id.train()
    for train_index, valid_index in kfold.split(train_dataset):
        train_subset = Subset(train_dataset, train_index)
        valid_subset = Subset(train_dataset, valid_index)

        # 創建新的 DataLoader 來處理這些子集
        train_loader = DataLoader(train_subset, batch_size=4, shuffle=True, drop_last=True)
        valid_loader = DataLoader(valid_subset, batch_size=4, shuffle=False, drop_last=True)
    with tqdm(train_loader, unit="batch", desc=f"Epoch {epoch + 1}") as tepoch:
        for batch in tepoch:
            tepoch.set_description(f"Epoch {epoch}")
            input_neutral = batch['neutral']        
            input_smile = batch['smile']
            output_neutral, x_encoded = model_Enc(input_neutral)
            output_smile, y_encoded = model_Enc(input_smile)
            output_Id = model_Id(x_encoded, y_encoded)            
                        
            Id_loss = loss(output_Id, batch['same_id'])
            opt_Id.zero_grad()
            Id_loss.backward()
            opt_Id.step()

            probs = torch.sigmoid(output_Id) 
               
            predictions = (probs > 0.5).float()
            same_id_f1 = f1_score(batch['same_id'].cpu().numpy(), predictions.cpu().numpy())
            print(f"ID f1 score {same_id_f1}")
            
            tepoch.set_postfix(loss=Id_loss.item())
        model_Id.eval()

        with torch.no_grad():
            total_loss = 0
            for batch in valid_loader:
                input_neutral = batch['neutral']        
                input_smile = batch['smile']
                output_neutral, x_encoded = model_Enc(input_neutral)
                output_smile, y_encoded = model_Enc(input_smile)
                output_Id = model_Id(x_encoded, y_encoded)  
                
                Id_loss = loss(output_Id, batch['same_id'])
                total_loss += Id_loss

            total_loss = total_loss/(len(valid_loader))
            scheduler.step(total_loss)
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
            for batch in test_loader:
                input_neutral = batch['neutral']        
                input_smile = batch['smile']
                output_neutral, x_encoded = model_Enc(input_neutral)
                output_smile, y_encoded = model_Enc(input_smile)
                output_Id = model_Id(x_encoded, y_encoded) 
                probs = torch.sigmoid(output_Id) 
               
                predictions = (probs > 0.5).float()
                same_id_f1 = f1_score(batch['same_id'].cpu().numpy(), predictions.cpu().numpy())
                print(f"ID f1 score {same_id_f1}")
    if epoch % 500 == 0:
        torch.save(model_Id.state_dict(), f"{args.export_path}_epoch{epoch}.pth")
        print(f"successful save model")
                
                