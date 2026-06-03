"""Generate academically-sourced English benchmark subsets and Turkish translations.

Usage:
    python data/generate_benchmarks.py --output-dir data/processed --sample-size 500 --seed 42

This script requires internet access and will download datasets/models from HuggingFace.
Required packages: datasets, transformers, sentencepiece, tqdm
"""
from pathlib import Path
import json
import argparse
from datetime import datetime

from datasets import load_dataset
from transformers import pipeline
from tqdm.auto import tqdm


def default_datasets():    return [
        ("squad_v2", "qa", "squad_v2"),
        ("fever", "fact_verification", "fever"),
        ("natural_questions", "qa", "natural_questions"),
        ("xsum", "summarization", "xsum"),
    ]


def translate_texts(texts, translator, batch_size=16):
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        out = translator(batch)
        # pipeline returns list of dicts with 'translation_text'
        results.extend([o['translation_text'] for o in out])
    return results


def sample_and_save(hf_name, task, sample_size, out_dir, translator, seed, stream=False):
    if stream:
        ds_iter = load_dataset(hf_name, split='train', streaming=True)
        records_list = []
        for i, ex in enumerate(ds_iter):
            if i >= sample_size:
                break
            records_list.append(ex)
        ds = records_list
    else:
        for split_try in ('train', 'validation', 'test'):
            try:
                ds = load_dataset(hf_name, split=split_try)
                break
            except Exception:
                ds = None
        if ds is None:
            ds = load_dataset(hf_name)
        ds = ds.shuffle(seed=seed)
        ds = ds.select(range(min(sample_size, len(ds))))

    records = []
    to_translate = []

    for ex in ds:
        if task == 'qa' and 'question' in ex:
            context = ex.get('context') or ex.get('article') or ''
            question = ex.get('question')
            answers = ex.get('answers') or ex.get('answers', {})
            records.append({
                'context': context,
                'question': question,
                'answers': answers,
            })
            to_translate.extend([context, question])
        elif task == 'fact_verification':
            claim = ex.get('claim') or ex.get('statement') or ex.get('sentence') or ''
            label = ex.get('label') or ex.get('verifiable', '')
            records.append({'claim': claim, 'label': label})
            to_translate.append(claim)
        elif task == 'summarization':
            doc = ex.get('document') or ex.get('article') or ex.get('document') or ''
            summary = ex.get('summary') or ex.get('highlights') or ''
            records.append({'document': doc, 'summary': summary})
            to_translate.extend([doc, summary])
        else:

            text = ex.get('text') or json.dumps(ex)
            records.append({'text': text})
            to_translate.append(text)


    mapping = {}
    if translator is not None:
        unique_texts = list(dict.fromkeys([t for t in to_translate if t]))
        translated = []
        if unique_texts:
            translated = translate_texts(unique_texts, translator)
        mapping = {u: t for u, t in zip(unique_texts, translated)}


    out_dir.mkdir(parents=True, exist_ok=True)
    english_path = out_dir / 'data.jsonl'
    with english_path.open('w', encoding='utf-8') as fe:
        for e in records:
            fe.write(json.dumps(e, ensure_ascii=False) + '\n')


    if translator is not None:
        turkish_records = []
        for rec in records:
            tr = {}
            for k, v in rec.items():
                if isinstance(v, dict):
                    tr[k] = {}
                    for ik, iv in v.items():
                        if isinstance(iv, list):
                            tr[k][ik] = [mapping.get(s, '') for s in iv]
                        else:
                            tr[k][ik] = mapping.get(iv, '')
                elif isinstance(v, list):
                    tr[k] = [mapping.get(s, '') for s in v]
                else:
                    tr[k] = mapping.get(v, '')
            turkish_records.append(tr)
        turkish_path = out_dir / 'data_tr.jsonl'
        with turkish_path.open('w', encoding='utf-8') as ft:
            for t in turkish_records:
                ft.write(json.dumps(t, ensure_ascii=False) + '\n')


    info = {
        'citation': '',
        'description': f'Sampled {len(records)} examples from {hf_name} for task {task}.',
        'homepage': f'https://huggingface.co/datasets/{hf_name}',
        'license': '',
        'semantic_role': task,
        'sample_size': len(records),
        'seed': seed,
        'created_at': datetime.utcnow().isoformat() + 'Z'
    }
    with (out_dir / 'dataset_info.json').open('w', encoding='utf-8') as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    print(f'Wrote {len(records)} examples to {out_dir}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=str, default='data/processed')
    parser.add_argument('--sample-size', type=int, default=500)
    parser.add_argument('--no-translate', action='store_true', help='Do not perform translation; only save English subsets')
    parser.add_argument('--stream', action='store_true', help='Use streaming mode to avoid downloading full datasets to disk')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # prepare translator pipeline unless disabled
    translator = None
    if not args.no_translate:
        print('Loading translation model (Helsinki-NLP/opus-mt-en-tr)...')
        translator = pipeline('translation', model='Helsinki-NLP/opus-mt-en-tr', device=-1)

    datasets = default_datasets()
    for ds_name, task, hf_name in datasets:
        print('Processing', ds_name)
        out_dir = out_root / ds_name
        try:
            sample_and_save(hf_name, task, args.sample_size, out_dir, translator, args.seed, stream=args.stream)
        except Exception as e:
            print(f'Warning: failed to process {hf_name}: {e}')
            continue


if __name__ == '__main__':
    main()
