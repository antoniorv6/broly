from model import GetBROLYFirstStage
from DataGenerator import DataGen
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import random

def hamming_distance(a, b):
    return len([i for i in filter(lambda x: x[0] != x[1], zip(a, b))])

def test_BROLY_model(model, generator, epoch):
    global_accuracy = 0
    idx_report = random.randint(0, generator.get_val_size() - 1)
    for i in range(generator.get_val_size() // 16 ):
        X_source, X_target, Y = generator.val_batch(16)
        predictions = model.predict([X_source, X_target])
        predictions[predictions>=0.5] = 1
        predictions[predictions<0.5] = 0
        for idx, prediction in enumerate(predictions):
            global_accuracy += accuracy_score(Y[idx], prediction)
        
        if i == idx_report:
            print(f"BATCH: {idx_report} - CLASSIFICATION REPORT")
            print(classification_report(Y, predictions, labels=list(range(0, generator.get_val_size()-1))))   
        
    print(f"EPOCH: {epoch} ||  || GLOBAL ACCURACY: {global_accuracy / generator.get_val_size()} ||")
    
    

def main():
    model = GetBROLYFirstStage((224,224,3), (224,224,3), 32, 4, 8, [128, 64], [16, 8], 5)
    generator = DataGen()
    BATCH_SIZE = 16

    for EPOCH in range(10000):
        X_source, X_target, Y = generator.train_batch(BATCH_SIZE)
        model.fit([X_source, X_target], Y, verbose=0)
        if EPOCH % 10 == 0:
            test_BROLY_model(model, generator, EPOCH)


if __name__ == "__main__":
    main()