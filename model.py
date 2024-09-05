import torch
import torch.nn as nn
import timm

from pooling import GeM, ViTGeM

'''
SkinModel : EfficientNet에서만 사용하세요.
'''
class SkinModel(nn.Module):
    def __init__(self, model_name, num_classes=1, pretrained=True, checkpoint_path=None):
        super(SkinModel, self).__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained, 
                                       checkpoint_path=checkpoint_path)
        in_features = self.model.classifier.in_features
        self.num_classes = num_classes
        self.model.classifier = nn.Identity()
        self.model.global_pool = nn.Identity()
        self.pooling = GeM()
        self.linear = nn.Linear(in_features, num_classes)

        
    def forward(self, images):
        output = self.model(images)
        output=self.pooling(output).flatten(1)
        output = self.linear(output)
        return output
        
        
'''
ViTSkinModel : Vision Transformer를 사용하는 모델입니다. 
ViT인 경우 이 모델을 사용하셔야 합니다.
'''
class ViTSkinModel(nn.Module):
    def __init__(self, model_name,pretrained=True, checkpoint_path=None):
        super(ViTSkinModel, self).__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained, 
                                       checkpoint_path=checkpoint_path,
                                       global_pool='', num_classes=0)
        self.softmax = nn.Softmax()
        self.pooling = ViTGeM()
        self.linear = nn.Linear(self.model.head.in_features, 1)
            
    def forward(self, images):
        # cls 토큰 뽑아봄
        output = self.model(images)
        output = self.pooling(output.permute(0, 2, 1)).flatten(1)
        output = self.linear(output)
        
        # softmax는 훈련시에 안 먹고 추론시에 사용할 것.
        return output
            
'''
SkinCOnvNext : ConvNext 모델입니다.
'''
class SkinConvNext(nn.Module):
    def __init__(self, model_name, num_classes=1, pretrained=True, checkpoint_path=None):
        super(SkinConvNext, self).__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained, 
                                        checkpoint_path=checkpoint_path)
        self.num_classes = num_classes
        self.model.global_pool = nn.Identity()
        self.pooling = GeM()
        self.linear = nn.Linear(self.model.head.fc.in_features, num_classes)
    
    def forward(self, images):
        output = self.model(images)
        output=self.pooling(output).flatten(1)
        output = self.linear(output)
        return output
       
'''
MaxVit
'''
class SkinMaxVit(nn.Module):
    def __init__(self, model_name, num_classes=1, pretrained=True, checkpoint_path=None):
        super(SkinMaxVit, self).__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained, 
                                        checkpoint_path=checkpoint_path)
        self.num_classes = num_classes
        self.model.global_pool = nn.Identity()
        self.pooling = GeM()
        self.linear = nn.Linear(self.model.head.fc.in_features, num_classes)
    
    def forward(self, images):
        output = self.model(images)
        output=self.pooling(output).flatten(1)
        output = self.linear(output)
        return output
            
'''
SkinCoat
'''
class SkinCoat(nn.Module):
    def __init__(self, model_name, num_classes=1, pretrained=True, checkpoint_path=None):
        super(SkinCoat, self).__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained, 
                                        checkpoint_path=checkpoint_path, global_pool="", num_classes=0)
        self.num_classes = num_classes
        self.model.classifier = nn.Identity()
        self.model.global_pool = nn.Identity()
        self.pooling = GeM()
        self.linear = nn.Linear(self.model.fc2.out_features, num_classes)
    
    def forward(self, images):
        output = self.model(images)
        output=self.pooling(output).flatten(1)
        output = self.linear(output)
        return output

    

    
        
