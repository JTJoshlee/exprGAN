import torch
import torch.nn as nn
import torch.nn.functional as F

class TotalLoss(nn.Module):
    def __init__(self, args):
        super(TotalLoss, self).__init__()
        self.loss_expfn = ExpressionClassifyLoss(args)
        self.loss_Idfn = IdClassifyLoss(args)
        self.loss_attention = AttentionLoss(args)

    def forward(self, output_neutral, output_smile, output_Id, batch, com_neutral, com_smile):
        loss_express = self.loss_expfn(output_neutral, 'neutral') + self.loss_expfn(output_smile, 'smile')
        loss_Id = self.loss_Idfn(output_Id, batch['same_id'])  
        loss_attention_neutral = self.loss_attention(com_neutral, 'neutral')
        loss_attention_smile = self.loss_attention(com_smile, 'smile')
        total_loss = loss_express + loss_Id + loss_attention_neutral + loss_attention_smile

        return total_loss
class ExpressionClassifyLoss(nn.Module):
    def __init__(self,args):
        super(ExpressionClassifyLoss, self).__init__()        
        self.batch_size = args.batch_size
    def forward(self, predictions, targets):
        input_probalities = F.softmax(predictions, dim=1)
                
        print(input_probalities)
        entropy = nn.CrossEntropyLoss()
        if targets == 'neutral':
            target = torch.tensor([0,0]).to('cuda')
        else: 
            target = torch.tensor([1,1]).to('cuda')
        output = entropy(predictions,target)        
        return output
    
class IdClassifyLoss(nn.Module):
    def __init__(self,args):
        super(IdClassifyLoss, self).__init__()
        self.batch_size = args.batch_size

    def forward(self, abs_id, target):
        entropy = nn.BCEWithLogitsLoss()     
        print(f"abs_id :{abs_id}")
        print(f"target: {target}")
        output = abs_id
        target_tensor = target       
        output = entropy(output, target_tensor)
               
        return output
    


class AttentionLoss(nn.Module):
    def __init__(self,args):
        super(AttentionLoss, self).__init__()
        self.batch_size = args.batch_size

    def forward(self, complentary_img_pred, targets):
        probabilities = F.softmax(complentary_img_pred, dim=1)
        
        if targets == 'neutral':            
            target_prob = probabilities[:,0]            
            
        else:
            target_prob = probabilities[:,1]
            

        return target_prob.mean()
        
        