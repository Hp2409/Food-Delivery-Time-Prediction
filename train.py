"""train.py
Training pipeline (generic scikit-learn style).

Usage:
    python train.py --data data/train.csv --target target_column --model_out models/model.joblib
"""
import argparse
import os
from preprocess import load_data, clean_data, split_data
from utils import save_model, evaluate_regression
from sklearn.ensemble import RandomForestRegressor
import joblib

def train(args):
    print('Loading data...')
    df = load_data(args.data_path, args.data_file)
    print('Cleaning data...')
    df = clean_data(df)
    print('Splitting...')
    X_train, X_test, y_train, y_test = split_data(df, args.target, test_size=args.test_size)
    print('Training model...')
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print('Evaluating...')
    evaluate_regression(model, X_test, y_test)
    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
    save_model(model, args.model_out)
    print(f'Model saved to {args.model_out}')

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data_path', default='data', help='Path to data folder')
    p.add_argument('--data_file', default='train.csv', help='Data file name inside data/')
    p.add_argument('--target', required=True, help='Name of the target column')
    p.add_argument('--model_out', default='models/model.joblib', help='Output model path')
    p.add_argument('--test_size', type=float, default=0.2)
    args = p.parse_args()
    train(args)
