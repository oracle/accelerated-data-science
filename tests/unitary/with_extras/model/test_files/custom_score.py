# THIS IS A CUSTOM SCORE.PY

model_name = "model.pkl"


def load_model(model_file_name=model_name):
    return model_file_name


def predict(data, model=load_model()):
    return {"prediction": "This is a custom score.py."}
