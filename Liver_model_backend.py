import joblib
from fastapi import FastAPI
from pydantic import BaseModel
import sklearn
import pandas as pd 
import shap
app = FastAPI()
sklearn.set_config(transform_output="pandas")
model = joblib.load(r"C:\FrostByte Project\Liver disease prediction model(enhanced+calibrated(sigmoid)).joblib")
rf_pipeline = model.calibrated_classifiers_[0].estimator
class InputData(BaseModel):
    Age:int
    Gender:str
    Total_Bilirubin:float
    Direct_Bilirubin:float
    Alkphos_Alkaline_Phosphotase:float
    Sgpt_Alamine_Aminotransferase:float
    Sgot_Aspartate_Aminotransferase:float
    Total_Protiens:float
    ALB_Albumin:float
    AG_Ratio_Albumin_Globulin_Ratio:float
def Dframe_Convertor(df , Transformer_name):
    return pd.DataFrame(df , columns = rf_pipeline.named_steps[Transformer_name].get_feature_names_out())
def Shap_explainations(df):
    df = rf_pipeline.named_steps["Imp_CT"].transform(df)
    df = Dframe_Convertor(df , "Imp_CT")
    df = rf_pipeline.named_steps["Scaler"].transform(df)
    df = Dframe_Convertor(df , "Scaler")
    df = rf_pipeline.named_steps["Encod"].transform(df)
    df = Dframe_Convertor(df , "Encod")
    df = rf_pipeline.named_steps["Tform"].transform(df)
    df = Dframe_Convertor(df , "Tform")
    explainer = shap.TreeExplainer(rf_pipeline.named_steps["LR"])
    shap_values = explainer(df).values[0 , : , 1]
    Index = rf_pipeline.named_steps['Tform'].get_feature_names_out()
    Clean_Index = []
    for items in Index:
        Clean_Index.append(items.split("__")[-1])
    impt = pd.Series(shap_values , index = Clean_Index)
    top_ind = impt.abs().sort_values(ascending = False).head(3)
    return top_ind
@app.post("/predict")
def Predictor(data: InputData):
    df = {
    "Age of the patient":data.Age, "Gender of the patient": data.Gender, 
    'Total Bilirubin':data.Total_Bilirubin,
       'Direct Bilirubin':data.Direct_Bilirubin, '\xa0Alkphos Alkaline Phosphotase':data.Alkphos_Alkaline_Phosphotase,
       '\xa0Sgpt Alamine Aminotransferase':data.Sgpt_Alamine_Aminotransferase, 'Sgot Aspartate Aminotransferase':data.Sgot_Aspartate_Aminotransferase ,
       'Total Protiens':data.Total_Protiens, '\xa0ALB Albumin':data.ALB_Albumin ,
       'A/G Ratio Albumin and Globulin Ratio': data.AG_Ratio_Albumin_Globulin_Ratio
    }
    columns = joblib.load("columns.pkl")
    df = pd.DataFrame([df] , columns = columns)
    Exps = Shap_explainations(df)
    sklearn.set_config(transform_output = "pandas")
    h_level = model.predict(df)
    response = model.predict_proba(df)[0][1]*100
    return {"Features responsible for this output": Exps , "Chances of Liver disease": f"{round(response , 2)}%" , "Risk Level":["Low" if h_level == 1 else "High"][0]}