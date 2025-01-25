import cv2
import torch
import os
import yaml
import torch.nn as nn
import ssl
import pretrainedmodels

class SEResNeXtFineTuner:
    def __init__(self, model_name='se_resnext50_32x4d', device="cuda"):
        self.dir_path = os.path.dirname(os.path.realpath(__file__))

        with open(os.path.join(self.dir_path, 'hyperparameters.yaml')) as f:
            self.args = yaml.safe_load(f)



        self.device = device
        self.model_name = model_name
        self.resize_shape = (224, 224) 

        original_context = ssl._create_default_https_context
        ssl._create_default_https_context = ssl._create_unverified_context
        self.model = pretrainedmodels.__dict__[model_name](pretrained='imagenet')
        ssl._create_default_https_context = original_context
        
        self.model.last_linear = nn.Linear(self.model.last_linear.in_features, len(self.args["classes"]))
        self.model = self.model.to(self.device)
        
        self.mean = self.model.mean
        self.std = self.model.std
    
    def freeze_layers(self):
        for param in self.model.parameters():
            param.requires_grad = False

        conv_count = 0
        for name, module in reversed(list(self.model.named_modules())):
            if isinstance(module, (nn.Conv2d, nn.ReLU, nn.BatchNorm2d, nn.Dropout)):
                conv_count += 1
                if conv_count <= 30:
                    for param in module.parameters():
                        param.requires_grad = True
    
    def predict(self, image, threshold=0.5):
        self.model.eval()

        image_resized = cv2.resize(image, self.resize_shape)        
        image_resized = image_resized / 255.0
        image_tensor = torch.tensor(image_resized, dtype=torch.float32).permute(2, 0, 1)

        image_tensor = (image_tensor - torch.tensor(self.mean).view(3, 1, 1)) / torch.tensor(self.std).view(3, 1, 1)

        image_tensor = image_tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(image_tensor)
            probs = torch.sigmoid(logits)
            preds = (probs > threshold).int().cpu().numpy()

        class_names = self.args['classes']
        predicted_classes = [class_names[i] for i, val in enumerate(preds[0]) if val == 1]
        return predicted_classes

    def load(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        path = os.path.join(dir_path, self.args["path"])

        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state['model_state_dict'])
        self.model_name = state['model_name']
        self.resize_shape = state['resize_shape']
        self.mean = state['mean']
        self.std = state['std']