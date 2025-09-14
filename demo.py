"""demo.py
Demo / inference script. Loads saved model and runs predictions on sample input or test file.

Usage:
    python demo.py --model models/model.joblib --data data/test.csv --output results.csv
"""
import argparse
import os
import pandas as pd
from utils import load_model, postprocess_predictions

def run_demo(args):
    print('Loading model...')
    model = load_model(args.model)
    if args.data_file and os.path.exists(os.path.join(args.data_path, args.data_file)):
        df = pd.read_csv(os.path.join(args.data_path, args.data_file))
        X = df.drop(columns=[args.target]) if args.target in df.columns else df
        print('Predicting...')
        preds = model.predict(X)
        out_df = pd.DataFrame({'prediction': preds})
        os.makedirs('results', exist_ok=True)
        out_path = args.output if args.output else os.path.join('results', 'predictions.csv')
        out_df.to_csv(out_path, index=False)
        print(f'Predictions saved to {out_path}')
    else:
        # simple manual demo
        sample = args.sample.split(',') if args.sample else []
        if not sample:
            print('No data provided. Provide --data_file or --sample for a quick demo.')
            return
        sample = [float(x) for x in sample]
        preds = model.predict([sample])
        print('Sample input:', sample)
        print('Prediction:', preds[0])

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='models/model.joblib', help='Path to saved model')
    p.add_argument('--data_path', default='data', help='Path to data folder')
    p.add_argument('--data_file', default='test.csv', help='Data file name for predictions')
    p.add_argument('--output', default=None, help='Output CSV path for predictions')
    p.add_argument('--target', default=None, help='If data_file contains target column name, provide it to drop before predicting')
    p.add_argument('--sample', default='', help='Comma-separated feature values for a quick prediction (e.g. "1.0,2.5,3.3")')
    args = p.parse_args()
    run_demo(args)
