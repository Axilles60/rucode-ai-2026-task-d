# Обучение итоговой модели (задача D, «Клиент под защитой»).
#
# Модель = 0.896 * ансамбль из 46 бустингов (ensemble.json) + 0.104 * gbdt4.
# Почему именно так, написано в README.md. Здесь ничего не подбирается: состав,
# гиперпараметры и веса зафиксированы, обучение идёт на всём hard_train.csv.
#
# Запуск: python train.py   (модели и метаданные пишутся в model/)
import json
import os
import pickle
import platform
import time

import catboost
import interpret
import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
from catboost import CatBoostRegressor

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, 'data')
MODEL_DIR = os.environ.get('MODEL_DIR', os.path.join(ROOT, 'model'))
THREADS = int(os.environ.get('THREADS', '2'))

SEED = 42          # все зёрна ниже получаются из него; оно же в имени файла ответа
N_SEEDS = 30       # сколько моделей с разными зёрнами усредняется в каждом члене ансамбля
BLEND_W = 0.104    # доля gbdt4 в итоговом прогнозе

TARGET, ID = 'protection_score', 'customer_id'
CAT_COLS = ['gender', 'region', 'city_type', 'education', 'family_status', 'employment', 'wealth_segment']


# --- подготовка признаков (predict.py импортирует отсюда, чтобы не было расхождений) ---

def frame_base(df, columns, levels):
    # категории -> pandas.Categorical с уровнями из train, числа как есть
    out = df[columns].copy()
    for c in CAT_COLS:
        out[c] = pd.Categorical(out[c].fillna('__MISSING__').astype(str), categories=levels[c])
    return out


def frame_ord(df, columns, ord_levels):
    # категории -> порядковые коды, пропуски остаются NaN
    out = df[columns].copy()
    for c in CAT_COLS:
        codes = pd.Categorical(out[c], categories=ord_levels[c]).codes.astype(float)
        out[c] = np.where(codes < 0, np.nan, codes)
    return out


def frame_ordfill(df, columns, ord_levels):
    # порядковые коды, все пропуски (и в категориях, и в числах) -> -1
    out = df[columns].copy()
    for c in CAT_COLS:
        out[c] = pd.Categorical(out[c], categories=ord_levels[c]).codes
    return out.fillna(-1)


def frame_ohe(df, columns, ohe_columns):
    # one-hot, набор столбцов как при обучении
    return pd.get_dummies(df[columns], columns=CAT_COLS, dummy_na=False, dtype=float).reindex(
        columns=ohe_columns, fill_value=0.0)


def frame_fill0(df, columns, levels):
    # как base, но пропуски в числовых признаках -> 0
    out = frame_base(df, columns, levels)
    num = [c for c in columns if c not in CAT_COLS]
    out[num] = out[num].fillna(0)
    return out


def frame_catboost(df, columns):
    # для CatBoost: категории строками, у каждого числового признака флаг пропуска
    cols = {}
    for c in CAT_COLS:
        cols[c] = df[c].fillna('__MISSING__').astype(str)
    for c in columns:
        if c in CAT_COLS:
            continue
        v = pd.to_numeric(df[c], errors='coerce').astype(float).replace([np.inf, -np.inf], np.nan)
        cols[c] = v
        cols[c + '__missing'] = v.isna().astype('int8')
    return pd.DataFrame(cols, index=df.index)


def make_frame(kind, df, meta):
    if kind == 'base':
        return frame_base(df, meta['columns'], meta['levels'])
    if kind == 'ord':
        return frame_ord(df, meta['columns'], meta['ord_levels'])
    if kind == 'ordfill':
        return frame_ordfill(df, meta['columns'], meta['ord_levels'])
    if kind == 'ohe':
        return frame_ohe(df, meta['columns'], meta['ohe_columns'])
    if kind == 'fill0':
        return frame_fill0(df, meta['columns'], meta['levels'])
    if kind == 'catboost':
        return frame_catboost(df, meta['columns'])
    raise ValueError(f'неизвестный тип признаков: {kind}')


# --- основной ансамбль ---

def is_deterministic(member):
    # без подвыборки строк и признаков результат от зерна не зависит, хватит одной модели
    p = member['params']
    return p.get('subsample', 1.0) == 1.0 and p.get('colsample_bytree', 1.0) == 1.0


def seeds_for(i, member):
    # Каждому члену -- зерно SEED плюс 29 своих зёрен. Если бы все члены брали одни и те же
    # зёрна, у них совпадали бы маски бэггинга и шум не усреднялся бы.
    # Модель с зерном SEED получает отдельный вес w_seed (см. README, раздел про зерно 42).
    if is_deterministic(member):
        return [SEED], [1.0]
    own = [SEED + 1000 * (i + 1) + k for k in range(1, N_SEEDS)]
    w = member['w_seed']
    return [SEED] + own, [w] + [(1 - w) / len(own)] * len(own)


def make_model(member, seed):
    if member['lib'] == 'lightgbm':
        return lgb.LGBMRegressor(n_estimators=member['n_estimators'], random_state=seed, verbosity=-1,
                                 n_jobs=THREADS, **member['params'])
    if member['lib'] == 'xgboost':
        return xgb.XGBRegressor(n_estimators=member['n_estimators'], random_state=seed, n_jobs=THREADS,
                                **member['params'])
    raise ValueError(member['lib'])


def save_model(model, lib, name):
    path = os.path.join(MODEL_DIR, name)
    if lib == 'lightgbm':
        model.booster_.save_model(path)
    elif lib in ('xgboost', 'catboost'):
        model.save_model(path)
    else:
        with open(path, 'wb') as f:
            pickle.dump(model, f)
    return name


def count_params(model, lib):
    if lib == 'lightgbm':
        return sum(2 * t['num_leaves'] - 1 for t in model.booster_.dump_model()['tree_info'])
    if lib == 'xgboost':
        return sum(len(t.strip().split('\n')) for t in model.get_booster().get_dump())
    return 0


# --- добавка gbdt4: CatBoost + LightGBM + XGBoost + EBM, все с зерном SEED ---

GBDT4 = [
    {'name': 'gbdt4_cb', 'lib': 'catboost', 'features': 'catboost',
     'params': dict(iterations=1014, depth=6, learning_rate=0.015625385938113703, l2_leaf_reg=1.0815490652457276,
                    random_strength=0.16529767441158863, bootstrap_type='Bernoulli', subsample=0.7732255381476606,
                    rsm=0.8776218546464724, border_count=254, nan_mode='Min', loss_function='RMSE')},
    {'name': 'gbdt4_lgbm', 'lib': 'lightgbm', 'features': 'base',
     'params': dict(n_estimators=453, objective='regression', learning_rate=0.02, num_leaves=31, min_child_samples=50,
                    subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0)},
    {'name': 'gbdt4_xgb', 'lib': 'xgboost', 'features': 'base',
     'params': dict(n_estimators=518, tree_method='hist', max_depth=6, learning_rate=0.02, subsample=0.8,
                    colsample_bytree=0.8, min_child_weight=5, reg_lambda=1.0, enable_categorical=True,
                    max_cat_to_onehot=1)},
    {'name': 'gbdt4_ebm', 'lib': 'ebm', 'features': 'base', 'params': dict(interactions=10, outer_bags=8)},
]
EXT = {'lightgbm': 'txt', 'xgboost': 'json', 'catboost': 'cbm', 'ebm': 'pkl'}


def fit_gbdt4_member(member, frame, y):
    p = member['params']
    if member['lib'] == 'catboost':
        model = CatBoostRegressor(random_seed=SEED, thread_count=THREADS, verbose=False, allow_writing_files=False,
                                  task_type='CPU', **p)
        return model.fit(frame, y, cat_features=CAT_COLS)
    if member['lib'] == 'lightgbm':
        return lgb.LGBMRegressor(random_state=SEED, verbosity=-1, n_jobs=THREADS, **p).fit(frame, y)
    if member['lib'] == 'xgboost':
        return xgb.XGBRegressor(random_state=SEED, n_jobs=THREADS, **p).fit(frame, y)

    from interpret.glassbox import ExplainableBoostingRegressor
    model = ExplainableBoostingRegressor(random_state=SEED, n_jobs=THREADS, **p)
    # EBM чуть-чуть (~1e-5) зависит от набора SIMD-инструкций процессора; фиксирую AVX2,
    # если интерфейс доступен, чтобы результат совпадал на разных машинах
    previous = None
    try:
        from interpret import develop
        from interpret.utils._native import Native
        previous = develop.get_option('acceleration')
        develop.set_option('acceleration', Native.AccelerationFlags_AVX2)
    except Exception:
        previous = None
    try:
        return model.fit(frame, y)
    finally:
        if previous is not None:
            develop.set_option('acceleration', previous)


def main():
    np.random.seed(SEED)
    t0 = time.time()
    train = pd.read_csv(os.path.join(DATA_DIR, 'hard_train.csv'))
    y = train[TARGET].to_numpy()
    X = train.drop(columns=[ID, TARGET])

    # всё, что нужно predict.py для тех же преобразований признаков
    meta = {
        'columns': list(X.columns),
        'levels': {c: sorted(X[c].fillna('__MISSING__').astype(str).unique()) for c in CAT_COLS},
        'ord_levels': {c: sorted(X[c].dropna().unique().tolist()) for c in CAT_COLS},
    }
    meta['ohe_columns'] = list(pd.get_dummies(X, columns=CAT_COLS, dummy_na=False, dtype=float).columns)

    with open(os.path.join(ROOT, 'ensemble.json'), encoding='utf-8') as f:
        members = json.load(f)
    os.makedirs(MODEL_DIR, exist_ok=True)

    total_params = 0
    for i, member in enumerate(members):
        frame = make_frame(member['features'], X, meta)
        seeds, weights = seeds_for(i, member)
        files = []
        for seed in seeds:
            model = make_model(member, seed).fit(frame, y)
            files.append(save_model(model, member['lib'], f'm{i:02d}_s{seed}_{member["name"]}.{EXT[member["lib"]]}'))
            total_params += count_params(model, member['lib'])
        member['files'], member['file_weights'] = files, weights
        print(f'{i:2d} {member["name"]:40s} моделей {len(files):2d}  ({time.time() - t0:.0f} c)', flush=True)

    blend = []
    for member in GBDT4:
        model = fit_gbdt4_member(member, make_frame(member['features'], X, meta), y)
        name = save_model(model, member['lib'], f'{member["name"]}_s{SEED}.{EXT[member["lib"]]}')
        blend.append(dict(member, file=name))
        print(f'   {member["name"]}  ({time.time() - t0:.0f} c)', flush=True)

    versions = {'catboost': catboost.__version__, 'lightgbm': lgb.__version__, 'xgboost': xgb.__version__,
                'interpret': interpret.__version__, 'numpy': np.__version__, 'pandas': pd.__version__,
                'scikit-learn': sklearn.__version__}
    manifest = {'seed': SEED, 'n_seeds': N_SEEDS, 'members': members,
                'blend': {'weight': BLEND_W, 'members': blend},
                'n_params_main': int(total_params), 'python': platform.python_version(), 'versions': versions, **meta}
    with open(os.path.join(MODEL_DIR, 'manifest.json'), 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    print(f'готово за {time.time() - t0:.0f} c, модели в {MODEL_DIR}')


if __name__ == '__main__':
    main()
