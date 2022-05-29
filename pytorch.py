# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor

import numpy as np
import pandas as pd
import geopandas as gpd

import seaborn as sns
# -

gdf = gpd.read_file('areas_rides_pois.geojson')

dr_features = """amenity-restaurant 
leisure-pitch 
amenity-fast_food 
amenity-place_of_worship    
leisure-garden               
leisure-park                 
amenity-cafe                 
amenity-school               
shop-convenience             
leisure-playground           
shop-clothes                 
amenity-bar                  
amenity-bank                 
shop-hairdresser             
shop-supermarket             
amenity-pharmacy             
shop-beauty                  
amenity-bicycle_rental       
shop-deli                     
shop-laundry                  
leisure-swimming_pool         
shop-alcohol                  
tourism-hotel               
amenity-fuel                
shop-bakery 
tourism-artwork
leisure-fitness_centre      
shop-dry_cleaning           
office-company              
shop-car_repair             
amenity-dentist             
tourism-information         
amenity-pub                 
amenity-clinic              
amenity-vending_machine     
amenity-doctors             
shop-variety_store          
office-estate_agent         
amenity-library             
amenity-post_office         
tourism-attraction          
office-diplomatic           
shop-gift                   
shop-jewelry                
leisure-track               
shop-vacant""".split('\n')
dr_features = [s.strip() for s in dr_features]


class PandasDataset(Dataset):
    def __init__(self, df, features, target_column, postfix="", standardize=True):
        if postfix != "":
            features = [f"{f}{postfix}" for f in features]
            
        for feature in features:
            if standardize:
                df[feature] = (df[feature] - df[feature].mean()) / (df[feature].var()*0.5)
                

        x=df.fillna(0).loc[:, features].values
        y=df.fillna(0).loc[:, target_column].values

        self.x_train=torch.tensor(x, dtype=torch.float32)
        self.y_train=torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y_train)
  
    def __getitem__(self,idx):
        return self.x_train[idx],self.y_train[idx]

# +
df = gdf
df = df[df['num_dropoffs_20']>10000]
#df = gdf[gdf['do_relative']<1]
#df = gdf[gdf['is_center']]

ds = PandasDataset(df, dr_features, 'do_relative')# , postfix='_normalized_w')
batch_size = 1

dataloader = DataLoader(ds, batch_size=batch_size)

for X, y in dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break
    
    
len(ds.x_train)

# +
# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(46, 20),
            nn.ReLU(),
            nn.Linear(20,2)
            #nn.Linear(512, 2),
            #nn.Linear(512, 2),
            #nn.ReLU(),
        )
        self.regress = nn.Sequential(
            #nn.ReLU(),
            #nn.Linear(2,512),
            #nn.ReLU(),
            nn.Linear(2,1))
        
        self.simple = nn.Sequential(nn.Linear(46,1))

    def forward(self, x):
        x = self.flatten(x)
        logits = self.regress(self.linear_relu_stack(x))[0]
        #logits = self.simple(x)[0]
        return logits

model = NeuralNetwork().to(device)
print(model)

loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ##if batch % 1000 == 0:
        #    loss, current = loss.item(), batch * len(X)
        #    print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            
            
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
    test_loss /= num_batches
    print(f"Avg loss: {test_loss:>8f} \n")
    return test_loss


# -

epochs = 500
losses = []
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(dataloader, model, loss_fn, optimizer)
    losses.append(test(dataloader, model, loss_fn))
print("Done!")

# +
l = []
model.eval()
with torch.no_grad():
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        twod = model.linear_relu_stack(X)
        pred = model(X)
        for el_pred, el_twod, el_y in zip(pred, twod, y):
            loss = loss_fn(el_pred, el_y)
            el_pred = el_pred.numpy()
            el_twod = el_twod.numpy()
            el_y = el_y.numpy()
            
            l.append({
                'prediction': el_pred,
                'a': el_twod[0],
                'b': el_twod[1],
                'y': el_y,
                'loss': loss.numpy()
                
            })

df = pd.DataFrame(l)
# -

print(len(df))
df.sort_values('y', ascending=False)

df

df[df['y']<1].plot.scatter(x='a',
                       y='b',
                      c='prediction', colormap='cool')

df[df['y']<1].plot.scatter(x='a',
                       y='b',
                      c='loss', colormap='cool')

df[df['y']<1].plot.scatter(x='a',
                       y='b',
                      c='y', colormap='cool')

gdf[gdf['is_center']]['do_relative'].mean()

gdf['do_relative'].mean()

# +
res = 100

l = []
model.eval()
with torch.no_grad():
    for i in range(res+1):
        a = (df['a'].max() - df['a'].min()) / res * i + df['a'].min()
        for j in range(100):
            b = (df['b'].max() - df['b'].min()) / res * j + df['b'].min()
            pred = model.regress(torch.Tensor(((a,b))))           
            l.append({
                'prediction': pred[0].numpy(),
                'a': a,
                'b': b,
            })

df2 = pd.DataFrame(l)
        
# -

df2

df2.plot.scatter(x='a', y='b', c='prediction', colormap='cool')

for name, param in model.named_parameters(): 
    if param.requires_grad: 
        print(name, param.data)
        x = param.data[0].numpy()
        y = param.data[1].numpy()
        break

list(model.named_parameters()) 

pd.DataFrame({'x': x, 'y': y}).plot.scatter(x='x', y='y')


