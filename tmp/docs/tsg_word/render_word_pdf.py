"""Use the packaged DOCX renderer with a Microsoft Word PDF export."""
import importlib.util
import sys
from pathlib import Path

docx, pdf, out = map(Path, sys.argv[1:4])
skill = Path(r"C:\Users\admin\.codex\plugins\cache\openai-primary-runtime\documents\26.1013.11914\skills\documents")
spec = importlib.util.spec_from_file_location("packaged_docx_renderer", skill / "render_docx.py")
renderer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(renderer)
renderer.convert_to_pdf = lambda *args, **kwargs: (str(pdf.resolve()), "Microsoft Word native PDF export")
sys.argv = [str(skill / "render_docx.py"), str(docx.resolve()),
            "--output_dir", str(out.resolve()), "--emit_pdf"]
renderer.main()
