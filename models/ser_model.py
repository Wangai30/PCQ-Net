"""
AIO -- All Model in One
"""
# import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

#from transformers import  Wav2Vec2Model
#from transformers import HubertModel
from transformers import WavLMModel
from models.ser_spec import SER_AlexNet
from models.fasternet import FasterNetT1
from models.fenge import group_aggregation_bridge
---------------------------------------------分为4层---------------------------------------------
class Ser_Model2(nn.Module):
    def __init__(self):
        super(Ser_Model2, self).__init__() 
        #网络
        self.newnet =FasterNetT1(drop_path=0.0)
        self.GAB1 = group_aggregation_bridge(32,16)
        self.GAB2 = group_aggregation_bridge(48,32)
        self.GAB3 = group_aggregation_bridge(64,48)
                
        #预训练
        self.wav2vec2_model = WavLMModel.from_pretrained("/home/wxc/CA-MSER-main-9/wavlm-base-plus")
        self.avg_pool1 = nn.AdaptiveAvgPool2d((127,191))
        self.avg_pool2 = nn.AdaptiveAvgPool2d((63,95))
        self.avg_pool3 = nn.AdaptiveAvgPool2d((31,47))
        #最后的池化与线性连接
        self.avg_pool = nn.AdaptiveAvgPool2d(1) #全局池化
        self.avg_pool22 = nn.AdaptiveAvgPool1d(128)
        self.post_dropout = nn.Dropout(p=0.1)
        self.post_linear1 = nn.Linear(160, 128) 
        self.post_linear2 = nn.Linear(768, 128)
       
        self.post_linear5 = nn.Linear(256, 149) 
        self.post_linear6 = nn.Linear(22080, 149) 
        self.post_linear7 = nn.Linear(256, 128) 
        self.post_linear8 = nn.Linear(128, 128) 
        self.post_linear9 = nn.Linear(128, 4) 

                                                                     
    def forward(self, audio_spec, audio_mfcc, audio_wav):      
        
        # audio_spec: [batch, 3, 256, 384]
        # audio_mfcc: [batch, 300, 40]
        # audio_wav: [32, 48000]
        
        x1, x2, x3, x4 = self.newnet(audio_spec) #audio_spec x4torch.Size([32, 256, 15, 23])
        
     #预训练     
        audio_wav_ = self.wav2vec2_model(audio_wav.cuda()).last_hidden_state # [baftch, 149, 768] 
        audio_wav = audio_wav_.reshape(audio_wav_.shape[0],1,149,768) # [batch, 1, 149,768]
        audio_wav1 = self.avg_pool1(audio_wav) #目标[ ,1, 127,191]
        audio_wav2 = self.avg_pool2(audio_wav1)#目标[ ,1, 63,95]
        audio_wav3 = self.avg_pool3(audio_wav2)#目标[ ,1, 31,47]
    
    #分割应用
        fusion1 = self.GAB1(x2, x1, audio_wav1)
        fusion1_avg = self.avg_pool(fusion1)
       
        fusion2 = self.GAB2(x3, x2, audio_wav2)
        fusion2_avg = self.avg_pool(fusion2)
        
        fusion3 = self.GAB3(x4, x3, audio_wav3)
        fusion3_avg = self.avg_pool(fusion3)
        
    #concat
        x4_= self.avg_pool(x4) #[b ,64, 1 ,1]
        fusion = torch.cat([fusion1_avg, fusion2_avg, fusion3_avg, x4_], dim=1)  # [batch, 160,1,1] 
        fusion = fusion.reshape(fusion.shape[0],-1) # [batch, 160]
        fusion_dropout = self.post_dropout(fusion)# [batch, 160] 
        fusion_linear = F.relu(self.post_linear1(fusion_dropout), inplace=False)# [batch,128]
        
    #最后的预训练注意力 
        x4 = torch.flatten(x4, 1) # [batch, 22080] 
        x4 = self.post_dropout(x4)# [batch, 22080] 
        x4 = F.relu(self.post_linear6(x4), inplace=False)# [batch,149] 
        x4 = x4.reshape(x4.shape[0], 1, -1)# [batch, 1, 149] 
        audio_wav_att = torch.matmul(x4, audio_wav_) # [batch, 1, 768] 
        audio_wav_att = audio_wav_att.reshape(audio_wav_att.shape[0], -1) # [batch, 768] 
        audio_wav_att = self.post_dropout(audio_wav_att) # [batch, 768] 
        #audio_wav_att = F.relu(self.post_linear2(audio_wav_att), inplace=False) # [batch, 128] 
        audio_wav_p = self.avg_pool22(audio_wav_att)
        
    #分类

        audio_att = torch.cat([fusion_linear, audio_wav_p], dim=-1)  # [batch, 256] 
        audio_att = self.post_dropout(audio_att) # [batch, 256] 
        audio_att_1 = F.relu(self.post_linear7(audio_att), inplace=False) # [batch, 128] 
        audio_att_1 = self.post_dropout(audio_att_1) # [batch, 128] 
        audio_att_2 = F.relu(self.post_linear8(audio_att_1), inplace=False)  # [batch, 128] 
        output_att = self.post_linear9(audio_att_2) # [batch, 4] 

  
        output = {
            'F1': audio_wav_p,
            'F2': audio_att_1,
            'F3': audio_att_2,
            'F4': output_att,
            'M': output_att
        }            
