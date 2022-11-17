from tensorflow import keras


class TrainNetwork():
    def __init__(self, model, model_name):
        self.model = model
        self.model_name = model_name

    def train_model(self, train_method):
        train_method(self.model)

    def save_model(self):
        self.model.save()


class PredictNetwork():
    def __init__(self, model_name):
        self.model = keras.models.load_model(model_name)

    def predict_from_image(self, predict_method, image_name):
        return predict_method(self.model, image_name)
