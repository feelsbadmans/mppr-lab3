import sys
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from src.predict_methods import mnist_img_to_str
from src.train_methods import mnist_train

from src.models import mnist_model
from src.network import TrainNetwork, PredictNetwork

def main():
    args = sys.argv[1:]
    
    try:
        if args[0] == 'train' and args[1]:
            network = TrainNetwork(mnist_model(), 'models/' + args[1] + '.h5')
            network.train_model(mnist_train)
            network.save_model()

        elif args[0] == 'predict' and args[1] and args[2]:
            network = PredictNetwork('models/' + args[1] + '.h5')
            print(network.predict_from_image(mnist_img_to_str, 'assets/' + args[2]))
    except KeyboardInterrupt:
        print("\nProcess stopped!")
    except:
        print("\nIncorrect args! Use 'train <model_name>' for train model or 'predict <model_name> <image_name>' for model's prediction!")


if __name__ == "__main__":
    main()
