"""Quick script to show table cell extraction on BM_002.png."""
import warnings, json, sys, os
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from akshardrishti.pipeline import AksharDrishtiPipeline
from akshardrishti.config import Config

pipe = AksharDrishtiPipeline(Config.load())
doc  = pipe.process('benchmark/images/BM_002.png')

for r in doc.pages[0].ordered_regions():
    if r.type.value == 'table':
        os.makedirs('outputs', exist_ok=True)
        result = {
            'n_cells': len(r.cells),
            'cells': [
                {'row': c.row, 'col': c.col, 'text': c.text}
                for c in sorted(r.cells, key=lambda c: (c.row, c.col))
            ],
            'compose_text': r.text,
        }
        with open('outputs/table_test.json', 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print('Written to outputs/table_test.json')
        print(f'Cells found: {len(r.cells)}')
        print()
        rows = {}
        for c in r.cells:
            rows.setdefault(c.row, []).append(c)
        for row_idx in sorted(rows):
            cells_in_row = sorted(rows[row_idx], key=lambda c: c.col)
            print(f'Row {row_idx}: ' + ' | '.join(c.text for c in cells_in_row))
