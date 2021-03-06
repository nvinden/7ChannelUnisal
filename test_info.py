#Path to the directory that contains your different global descriptor datasets
DATASET_PATH = "/content/drive/MyDrive/7Channel/Salicon"

#information about the file directories you want to use to train in 
#the DATA_SET path
#  dir: this is the name of the directory that the program will save
#       compiled pictures into, or grab them out of if they exist
# func: this is the name of the function in sevenchanneltrans.py that will
#       run if the image is not already found in the dataset folder. It will
#       save the picture for next time.
#  end: the picture format
# chan: the number of channels the image can produce, either 1 or 3
CHANNELS = (
    {"dir": "rgb", "func": "make_rgb_mean_layer", "end": "jpg", "chan": 3, "mean": [0.095, 0.098, 0.100],
    "std": [0.131, 0.143, 0.151]},
    {"dir": "dark", "func": "make_dark_channel", "end": "jpg", "chan": 1, "mean": [0.180,],
    "std": [0.180,]},
    {"dir": "depth_kitti", "func": "depth_kitti", "end": "png", "chan": 1, "mean": [0.9,],
    "std": [0.1,]},
    {"dir": "depth_nyu", "func": "depth_nyu", "end": "png", "chan": 1, "mean": [0.9,],
    "std": [0.1,]}
)

#Note: All datasets must be in the format DATASET_PATH/dir_name/(train test and val)