from model import GetBROLYFirstStage
from DataGenerator import DataGen
import numpy as np
import tqdm

def hamming_distance(a, b):
    return len([i for i in filter(lambda x: x[0] != x[1], zip(a, b))])

def test_BROLY_model(model, generator, epoch):
    total_hamming_distance = 0
    good = 0
    for _ in range(generator.get_val_size() // 16 ):
        X_source, X_target, Y = generator.val_batch(16)
        predictions = model.predict([X_source, X_target])
        predictions[predictions>=0.5] = 1
        predictions[predictions<0.5] = 0
        for idx, prediction in enumerate(predictions):
            total_hamming_distance += hamming_distance(prediction, Y[idx])
            for jdx, element in enumerate(prediction):
                if element == Y[idx, jdx]:
                    good += 1
        
    print(f"EPOCH: {epoch} || GLOBAL HAMMING DISTANCE: {total_hamming_distance} | MEAN HAMMING DISTANCE: {total_hamming_distance / generator.get_val_size()} | GLOBAL ACCURACY: {good / (generator.get_val_size() * 5)} ||")   
    

def main():
    model = GetBROLYFirstStage((224,224,3), (224,224,3), 64, [256, 128, 64], 8, 0, 12)
    generator = DataGen()
    BATCH_SIZE = 16

    for EPOCH in range(10000):
        X_source, X_target, Y = generator.train_batch(BATCH_SIZE)
        model.fit([X_source, X_target], Y, verbose=0)
        if EPOCH % 10 == 0:
            test_BROLY_model(model, generator, EPOCH)


if __name__ == "__main__":
    main()