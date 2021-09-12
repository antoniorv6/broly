import cv2
import os
import numpy as np
import tqdm
from sklearn.model_selection import train_test_split 

CONST_TEST_FOLDER = "Test/Generator/"

class DataGen:
    def __init__(self):
        images = os.listdir("Data")

        self.images_list, self.validation_list = train_test_split(images, test_size=0.25, shuffle=True, random_state=1)

        self.images_index = 0
        self.val_images_index = 0
        self.bitwise_methods = [self.flip_horizontal, 
                                self.flip_vertical, 
                                self.zoom, 
                                self.blur, 
                                self.invert]
    
    def read_image(self, idxToRead, imagelist):
        return cv2.imread(f"Data/{imagelist[idxToRead]}")
    
    def flip_horizontal(self,image):
        return cv2.flip(image, 0)

    def flip_vertical(self,image):
        return cv2.flip(image, 1)
    
    def zoom(self,image):
        image_size = image.shape
        scaled_up_image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
        half_image = int(image_size[1]/2)
        three_quarter = int(image_size[1]/2) + image_size[1]
        return scaled_up_image[half_image:three_quarter, half_image:three_quarter]
    
    def blur(self, image):
        return cv2.GaussianBlur(image, (7,7), 0)
    
    def invert(self, image):
        return 255. - image
    
    def reset(self):
        self.images_index = 0
    
    def train_batch(self, BATCH_SIZE):
        X_source = []
        X_target = []
        Y = []
        image = None
        for idx_read in range(self.images_index, self.images_index+BATCH_SIZE):
            image = self.read_image(idx_read, self.images_list)
            edited_image = image
            edition_process = [np.random.binomial(1,0.5) for _ in range(5)]
            for idx in range(len(edition_process)):
                if edition_process[idx]:
                    edited_image = self.bitwise_methods[idx](edited_image)
            
            X_source.append(image)
            X_target.append(edited_image)
            Y.append(edition_process)


        # Reset the index if we reach end of images list
        self.images_index += BATCH_SIZE
        if self.images_index + BATCH_SIZE > len(self.images_list):
            self.images_index = 0

        return np.array(X_source), np.array(X_target), np.array(Y)
    
    def val_batch(self, BATCH_SIZE):
        X_source = []
        X_target = []
        Y = []
        image = None
        ## READ BATCH SIZE IMAGES
        for idx_read in range(self.val_images_index, self.val_images_index+BATCH_SIZE):
            image = self.read_image(idx_read, self.validation_list)
            edited_image = image
            edition_process = [np.random.binomial(1,0.5) for _ in range(5)]
            for idx in range(len(edition_process)):
                if edition_process[idx]:
                    edited_image = self.bitwise_methods[idx](edited_image)
            
            X_source.append(image)
            X_target.append(edited_image)
            Y.append(edition_process)

        # PERFORM RANDOMLY THE FOUR OPERATIONS, EDIT THE IMAGE AND SET BITWISE RESULT
        # ADD TO THE X AND Y ARRAYS

        # Reset the index if we reach end of images list
        self.val_images_index += BATCH_SIZE
        if self.val_images_index + BATCH_SIZE > len(self.validation_list):
            self.val_images_index = 0

        return np.array(X_source), np.array(X_target), np.array(Y)

    def get_val_size(self):
        return len(self.validation_list)

def main():
    dataGen = DataGen()
    BATCH_SIZE = 16
    dataGen.reset()
    X_source = [] 
    X_target = [] 
    Y = []
    for _ in tqdm.tqdm(range(1157)):
        X_source, X_target, Y = dataGen.train_batch(BATCH_SIZE)
    
    print("WRITING IMAGES FOR TEST")
    for i,element in enumerate(Y):
        imageName = ' '.join([str(elem) for elem in element]) + f"_{i}"
        cv2.imwrite(CONST_TEST_FOLDER + imageName + "_src.jpg", X_source[i])
        cv2.imwrite(CONST_TEST_FOLDER + imageName + "_tar.jpg", X_target[i])
    print("IMAGES WRITTEN")
    
    pass
    

if __name__ == "__main__":
    main()