"""Generate the seven source summary rows used by current eight-row Table V."""
from pathlib import Path
from insb_figures import write_table8
P=Path(__file__).resolve().parents[1]
write_table8()
for lang in ['en','zh']:
    (P/f'tables/table8_{lang}.tex').unlink()
