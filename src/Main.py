from tkinter import image_names
import pandas as pd
import numpy as np
from skimage import data, io
from PIL import Image

from auto_GA import automated_GA
from auto_GE import automated_GE
from auto_SA import automated_SA
from auto_Tabu import automated_Tabu

def main():
    
    test_images = [
        './image_segmentation/images/baboon.png',
        './image_segmentation/images/Lenna.png',
        './image_segmentation/images/pepper.png',
        './image_segmentation/images/plane.png',
    ]
    
    save_to = [
        'baboon.csv',
        'Lenna.csv',
        'pepper.csv',
        'plane.csv',
    ]

    for image,saver in zip(test_images,save_to):
        print(image)
        img2 = np.array(Image.fromarray(io.imread(image)).convert('L'))
        hist_, bins = np.histogram(img2, bins=range(256), density=False)
    
        data = []
        for threshold in range(1,8):

            #config, individual, fitness, thresh = automated_GA( hist=hist_, num_thresh=threshold)
            #data.append( ["GA", threshold, config, fitness, thresh ] )

            config, individual, fitness, thresh = automated_GE( hist=hist_, num_thresh=threshold)
            data.append( ["GE",threshold, config, fitness, thresh ] )

            ''''
            config, individual, fitness, thresh = automated_SA( hist=hist_, num_thresholds=threshold)
            data.append( ["SA",threshold,{},25, [i+1 for i in range(threshold)] ] )

            config, individual, fitness, thresh = automated_Tabu( hist=hist_, num_thresholds=threshold)
            data.append( ["TS",threshold,{},25, [i+1 for i in range(threshold)] ] )
            '''
        df = pd.DataFrame(data, columns=['Algorithm', 'num_threshold','config','fitness','thresholds'])
        df.to_csv(saver)

if __name__ == "__main__":
    main()