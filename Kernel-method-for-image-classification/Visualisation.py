import numpy as np
import matplotlib.pyplot as plt
import numpy as np

def Image_reconstructor(data):

    nb, N = data.shape
    red_channel = data[:,:N//3]
    min_red = red_channel.min(1).reshape(-1,1)
    max_red = red_channel.max(1).reshape(-1,1)
    red_channel = (red_channel-min_red)/(max_red-min_red)

    green_channel = data[:,N//3:2*N//3]
    min_green = green_channel.min(1).reshape(-1,1)
    max_green = green_channel.max(1).reshape(-1,1)
    green_channel = (green_channel-min_green)/(max_green-min_green)

    blue_channel = data[:,2*N//3:]
    min_blue = blue_channel.min(1).reshape(-1,1)
    max_blue = blue_channel.max(1).reshape(-1,1)
    blue_channel = (blue_channel-min_blue)/(max_blue-min_blue)

    data = np.hstack((red_channel,green_channel,blue_channel))
    data = data.reshape(nb,3,32,32).transpose(0,2,3,1)
    return data

def plot_image(train_set):
    
    plt.figure(figsize=(4*6, 4))
    for i in range(6):
        plt.subplot(1, 6, i+1)
        plt.imshow(train_set[i])
        plt.axis('off')  
    plt.show()


