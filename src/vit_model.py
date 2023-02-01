from torch_geometric.nn import GATConv, to_hetero
from torch.nn import Linear
import torch.nn.functional as F
from torch import nn
import torch
from timm import create_model
from torch_geometric.nn import HeteroConv, GATConv


class ViT(torch.nn.Module):
    def __init__(self, data, device=torch.device('cuda:0')):
        super().__init__()
        self.vit = create_model("vit_base_patch16_224", pretrained=True)
        self.vit.reset_classifier(num_classes=0)
        
        # add final multi-head
        out_channels_style = data['style'].x.shape[0]
        self.style_head = torch.nn.Linear(768, out_channels_style)

        out_channels_genre = data['genre'].x.shape[0]
        self.genre_head = torch.nn.Linear(768, out_channels_genre)

        out_channels_emotion = data['emotion'].x.shape[0]
        self.emotion_head = torch.nn.Linear(768, out_channels_emotion)        
        
    def forward(self, x): #, x_dict, edge_index_dict

    # la x sono le immagini sottoforma di tensori, devo fare in modo che mi parta sia l'addestramento delle features che 
    # quello della head grazie a queste immagini, e mi ritorni le tre predizioni
        print(x.shape)
        visual_features=self.vit(x)
        token_style=self.style_head(visual_features)    # patches
        token_genre= self.genre_head(visual_features)   # patches
        token_emotion=self.emotion_head(visual_features)#patches

        return token_style, token_genre, token_emotion
