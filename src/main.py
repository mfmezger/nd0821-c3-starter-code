# Put the code for your API here.
from fastapi import FastAPI
from pydantic import BaseModel, Field
from pathlib import Path
import pickle
import pandas as pd

app = FastAPI()

class ModelInput(BaseModel):
    # Using the first row of census.csv as sample
    age: int = Field(None, example=39)
    workclass: str = Field(None, example='State-gov')
    fnlgt: int = Field(None, example=77516)
    education: str = Field(None, example='Bachelors')
    education_num: int = Field(None, example=13)
    marital_status: str = Field(None, example='Never-married')
    occupation: str = Field(None, example='Adm-clerical')
    relationship: str = Field(None, example='Not-in-family')
    race: str = Field(None, example='White')
    sex: str = Field(None, example='Female')
    capital_gain: int = Field(None, example=2174)
    capital_loss: int = Field(None, example=0)
    hours_per_week: int = Field(None, example=40)
    native_country: str = Field(None, example='United-States')

@app.get("/")
async def root() -> dict:
    return {"message": "Welcome to the API!"}

async def get_model() -> tuple:
    model_path = Path(__file__).parent / "model" / "rf_model.pkl"
    encoder_path = Path(__file__).parent / "model" / "encoder.pkl"
    lb_path = Path(__file__).parent / "model" / "lb.pkl"

    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        with open(encoder_path, "rb") as f:
            encoder = pickle.load(f)
        with open(lb_path, "rb") as f:
            lb = pickle.load(f)
    except FileNotFoundError as e:
        print(f"Error loading model: {e}")
        model = None
        encoder = None
        lb = None
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None
        encoder = None
        lb = None

    return model, encoder, lb

@app.post("/predict")
async def predict(data: ModelInput):
    # unpack model dependencies
    model, encoder, lb = get_model()

    try:
        # format sample data for inference
        sample_dict = {key.replace('_', '-'): [value] for key, value in data.__dict__.items()}
        data = pd.DataFrame.from_dict(sample_dict)

        # perform inference
        cat_features = [
            "workclass",
            "education",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "native-country",
        ]

        X, _, _, _ = process_data(data, categorical_features=cat_features, label=None, 
            training=False, encoder=encoder, lb=lb)

        pred = inference(model, X)[0]

        # return prediction
        result = '<=50K' if pred == 0 else '>50K'
        return Response(content=result, media_type="text/plain")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))