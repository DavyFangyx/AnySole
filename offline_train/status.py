from pathlib import Path

root = Path(__file__).resolve().parent / 'queues'
for queue in sorted(root.glob('GPU*')):
    values = []
    for state in ('pending', 'running', 'done', 'failed'):
        values.append(f'{state}={len(list((queue / state).glob("*.conf")))}')
    print(queue.name, ' '.join(values))
