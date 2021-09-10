import cv2
import os
import numpy as np

class DataGenerator:
    def __init__(self):
        self.images_list = os.listdir("Data")
        self.dilationKernel = np.ones((5,5), np.uint8)
        self.images_index = 0
    
    def read_image(self, idxToRead):
        return cv2.imread(f"Data/{self.images_list[idxToRead]}")
    
    def flip_horizontal(self,image):
        return cv2.flip(image, 0)

    def flip_vertical(self,image):
        return cv2.flip(image, 1)
    
    def dilate(self,image):
        return cv2.erode(image, self.dilationKernel, iterations = 1)
    
    def invert(self, image):
        return cv2.invert(image)
    
    def __call__(self, BATCH_SIZE):
        X_source = []
        X_target = []
        Y = []
        image = None
        ## READ BATCH SIZE IMAGES
        for idx_read in range(self.images_index, self.images_index+BATCH_SIZE):
            image = self.read_image(idx_read)
            edition_process = [np.random.binomial(1,0.5) for _ in range(4)]
            
        

        # PERFORM RANDOMLY THE FOUR OPERATIONS, EDIT THE IMAGE AND SET BITWISE RESULT
        # ADD TO THE X AND Y ARRAYS

        # Reset the index if we reach end of images list
        self.images_index += BATCH_SIZE
        if self.images_index > len(self.images_list):
            self.images_index = 0

        pass

def main():
    pass

if __name__ == "__main__":
    main()