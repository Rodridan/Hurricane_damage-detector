from riskscope.model_pipeline import build_resnet_model

def test_build_model():
    model = build_resnet_model(input_shape=(224, 224, 3))
    model.summary()

if __name__ == "__main__":
    test_build_model()
