import joblib
import pandas as pd
from MEML.descriptor.descriptor import process_compositions

def process_data():
    """
    Process and merge the composition data and XRD data.
    """
    compositions_df, origin = process_compositions('element.xlsx', '../data/mp_data.xlsx')

    XRD = pd.read_csv('mp_XRD.csv')

    final_df = pd.merge(compositions_df.drop('Chemical_Formula', axis=1), XRD, on='index')
    final_df = pd.merge(origin.drop('composition', axis=1), final_df, on='index')

    final_df.to_csv('mp_all.csv', index=False)
    print("Data processed and saved successfully.")

def load_models():
    """
    Load pre-trained models.
    """
    models = {
        'semi': joblib.load('../model/semiconductor_model.pkl'),
        'stable': joblib.load('../model/stability_model.pkl'),
        'direct': joblib.load('../model/gap_type_model.pkl'),
    }
    return models

def predict_and_save(models):
    """
    Use the loaded models to make predictions and save the results to a CSV file.
    """
    # Read the data
    data = pd.read_csv('mp_all.csv')
    X_mp = data.iloc[:, 3:]

    predicts = []
    probs = []

    for model_name, model in models.items():
        predict = model.predict(X_mp)
        prob = model.predict_proba(X_mp)[:, 1]

        predicts.append(pd.DataFrame({f'predict_{model_name}': predict}))
        probs.append(pd.DataFrame({f'prob_{model_name}': prob}))

    mp_predict = pd.concat(predicts + probs, axis=1)

    mp_predict.to_csv('mp_predict.csv', index=False)
    print("Prediction completed and saved successfully.")

def main():
    """
    Main function to process the data, load models, and make predictions.
    """
    # Step 1: Process the data
    process_data()

    # Step 2: Load the models
    models = load_models()

    # Step 3: Perform predictions and save the results
    predict_and_save(models)

if __name__ == "__main__":
    main()
