import json
from pathlib import Path
from typing import Optional

def convert(in_path: str | Path, out_path: str | Path) -> int:
    in_path = Path(in_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with in_path.open('r', encoding='utf-8') as f_in, out_path.open('w', encoding='utf-8') as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            task_id = str(obj.get('id', f'ifeval_{n}'))
            instruction = obj.get('prompt') or obj.get('instruction') or obj.get('instructions')
            if instruction is None:
                instruction = json.dumps(obj, ensure_ascii=False)

            task = {
                'task_id': task_id,
                'dataset': 'ifeval',
                'task_type': 'instruction_following',
                'instruction': instruction,
                'input': obj.get('input', ''),
                'constraints': obj.get('constraints', []),
                'meta': obj,
            }
            f_out.write(json.dumps(task, ensure_ascii=False) + '\n')
            n += 1
    return n

def main():
    n = convert('data_raw/ifeval.jsonl', 'data_processed/tasks_ifeval.jsonl')
    print(f'Wrote {n} tasks -> data_processed/tasks_ifeval.jsonl')

if __name__ == '__main__':
    main()