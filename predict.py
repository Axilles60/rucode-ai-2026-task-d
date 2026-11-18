# Прогноз по сохранённым весам из model/ (без обучения) и запись submission_seed_{SEED}.csv.
#
#   python predict.py                         # закрытый тест data/hard_test_closed.csv -> submissions/
#   python predict.py --input data/hard_test.csv --output submissions/public/submission_seed_42.csv
import argparse
import json
import os
import pickle

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostRegressor

from train import ID, MODEL_DIR, TARGET, make_frame

ROOT = os.path.dirname(os.path.abspath(__file__))


def load(lib, path):
    if lib == 'lightgbm':
        return lgb.Booster(model_file=path)
    if lib == 'xgboost':
        m = xgb.XGBRegressor()
        m.load_model(path)
        return m
    if lib == 'catboost':
        m = CatBoostRegressor()
        m.load_model(path)
        return m
    with open(path, 'rb') as f:   # EBM сохранён через pickle
        return pickle.load(f)


def clipped(model, frame):
    return np.clip(np.asarray(model.predict(frame), dtype=float), 0.0, 100.0)


def predict(df, model_dir):
    with open(os.path.join(model_dir, 'manifest.json'), encoding='utf-8') as f:
        manifest = json.load(f)

    # основной ансамбль: прогноз члена -- взвешенное среднее его моделей по зёрнам,
    # итог -- простое среднее по членам
    total, n = 0.0, 0
    for member in manifest['members']:
        frame = make_frame(member['features'], df, manifest)
        p = sum(w * clipped(load(member['lib'], os.path.join(model_dir, f)), frame)
                for w, f in zip(member['file_weights'], member['files']))
        total = total + p
        n += 1
    pred = total / n

    # добавка gbdt4 с весом BLEND_W
    blend = manifest['blend']
    g = np.mean([clipped(load(m['lib'], os.path.join(model_dir, m['file'])), make_frame(m['features'], df, manifest))
                 for m in blend['members']], axis=0)
    pred = np.clip((1 - blend['weight']) * pred + blend['weight'] * g, 0.0, 100.0)
    return pred, manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', default=os.path.join(ROOT, 'data', 'hard_test_closed.csv'))
    ap.add_argument('--model-dir', default=MODEL_DIR)
    ap.add_argument('--output', default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    pred, manifest = predict(df, args.model_dir)
    out = args.output or os.path.join(ROOT, 'submissions', f'submission_seed_{manifest["seed"]}.csv')
    sub = pd.DataFrame({ID: df[ID], TARGET: pred})
    if sub.isna().any().any() or not sub[TARGET].between(0, 100).all():
        raise ValueError('В прогнозе или идентификаторах есть пропуски либо прогноз выходит за [0, 100]')
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    sub.to_csv(out, index=False)
    print(f'{len(sub)} строк -> {out} (среднее {pred.mean():.2f}, от {pred.min():.2f} до {pred.max():.2f})')


if __name__ == '__main__':
    main()
