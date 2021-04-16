

"""
    configuration class
"""


batch_size = 6
image_size = 256

epochs = 100
val_per_epochs = 10

save_path = "/root/sfwy/"

mode = "training"

train_path = "/root/sfwy/dataset/train"
val_path = "/root/sfwy/dataset/val"
test_path = "/root/sfwy/dataset/test"

generated_mask = True

if(generated_mask):
    train_mask_path = "/root/swy/dataset/mask/train"
    val_mask_path = "/root/sfwy/dataset/mask/val"