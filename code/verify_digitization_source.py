"""Check the cached Figure 4 raster against the supplied source PDF."""
from pathlib import Path
import json
from pypdf import PdfReader
from PIL import Image,ImageChops
P=Path(__file__).resolve().parents[1]
m=json.loads((P/'data/manifest.json').read_text())
cached=Image.open(P/m['source_raster']).convert('RGB')
page=PdfReader(P/m['source_pdf']).pages[m['source_pdf_page_index']]
assert any(im.image.size==cached.size and ImageChops.difference(im.image.convert('RGB'),cached).getbbox() is None for im in page.images)
print('Source PDF Figure 4 and cached digitization raster match exactly.')
