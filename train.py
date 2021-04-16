
"""
    @Author: Junjie Jin
    @Code: Junjie Jin
    @Description: train our model (Relying on our loader framework in https://github.com/sfwyly/loader)

"""

from config import *
from loader import *
from model import *
from tqdm import tqdm
from utils import *


def train():

    generator = build_model(mode = mode)

    for i in range(epochs):

        train_loss = trainer(generator)

        print(i," / ",epochs," train_loss: ",train_loss)
        if ((i + 1) % val_per_epochs == 0):
            save(i,generator)
            val_loss = validate()
            print(i, " / ", epochs, " val_loss: ", val_loss)
        log_save() # save log

def log_save():

    pass

def save(i,generator):
    generator.save_weights(save_path+str(i)+".h5")

def trainer(generator):

    train_dataloader = DataLoader(Dataset(root_path=train_path), batch_size=batch_size,
                                  image_size=(image_size, image_size), shuffle=True)
    val_dataloader = DataLoader(Dataset(root_path=train_path), batch_size=batch_size,
                                  image_size=(image_size, image_size), shuffle=True)

    if(generated_mask):

        train_mask_dataloader = DataLoader(Dataset(root_path = train_mask_path), batch_size=batch_size,
                                  image_size=(image_size, image_size), shuffle=True)
        val_mask_dataloader = DataLoader(Dataset(root_path = val_mask_path), batch_size=batch_size,
                                image_size=(image_size, image_size), shuffle=True)

    train_length = len(train_dataloader)

    all_loss = []

    for i,(X_trains,_) in enumerate(tqdm(train_dataloader)):

        if(not generated_mask):
            mask_list = getHoles((image_size,image_size),batch_size)
        else:
            length = len(train_mask_dataloader)
            mask_list = train_mask_dataloader[np.random.randint(length)][0]

        loss, style_loss, L1_loss, tvl_loss, perceptual_loss = train_step(generator, X_trains,X_trains * mask_list,mask_list)
        all_loss.append(loss.numpy())
    return np.mean(all_loss)

def validate():

    return 0


if(__name__=="__main__"):

    train()