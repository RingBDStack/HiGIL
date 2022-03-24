import json
import pickle
from pathlib import Path


def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as fin:
        obj = json.load(fin)
    return obj


def load_multiple_jsonl(file_paths):
    if isinstance(file_paths, (str, Path)):
        return _load_jsonl(file_paths)

    dps = []
    for file_path in file_paths:
        dps.extend(_load_jsonl(file_path))
    return dps


def peak_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as fin:
        obj = json.loads(fin.readline())
    return obj


def load_pickle(file_path):
    with open(file_path, 'rb') as fin:
        obj = pickle.load(fin)
    return obj


def save_json(obj, file_path):
    with open(file_path, 'w', encoding='utf-8') as fout:
        json.dump(obj, fout, indent=4, ensure_ascii=False)


def save_jsonl(objs, file_path):
    with open(file_path, 'w', encoding='utf-8') as fout:
        for idx, obj in enumerate(objs):
            fout.write(json.dumps(obj))
            if idx < len(objs) - 1:
                fout.write('\n')


def save_pickle(obj, file_path):
    with open(file_path, 'wb') as fout:
        pickle.dump(obj, fout)


def _load_jsonl(file_path):
    objs = []
    with open(file_path, 'r', encoding='utf-8') as fin:
        for line in fin:
            objs.append(json.loads(line))
    return objs
