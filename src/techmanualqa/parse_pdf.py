# 01
import pymupdf4llm
import sys
import yaml
from dotenv import load_dotenv
import logging
import json
import re
from pathlib import Path
import argparse  # For handling command-line arguments
import fitz  # from PyMuPDF – for page images
import statistics
import datetime

# --- Setup logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Load environment variables ---
load_dotenv()

# --- Load configuration ---
try:
    config_path = Path('config/settings.yaml')
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    logging.info("Configuration loaded successfully.")

    # Get new options from config, with defaults
    EXPORT_IMAGES = config.get('options', {}).get('export_images', True)
    IMAGE_SCALE = float(config.get('options', {}).get('image_scale', 2.0))
    IMG_SUBDIR = config.get('paths', {}).get('page_images_subdir', 'page_imgs')
    QC = config.get('qc_thresholds', {})

except FileNotFoundError:
    logging.error(f"Error: Configuration file not found at '{config_path}'. Please ensure it exists.")
    sys.exit(1)
except Exception as e:
    logging.error(f"Error loading configuration: {e}")
    sys.exit(1)

# --- Get paths from config ---
PDF_DIR = Path(config['paths']['pdf_directory'])
PROCESSED_DATA_DIR = Path(config['paths']['processed_data_dir'])


def get_language_from_filename(filename: str) -> str:
    """Extracts language (DE or EN) from filename based on common patterns."""
    if re.search(r'_DE_', filename, re.IGNORECASE):
        return "de"
    elif re.search(r'_EN_', filename, re.IGNORECASE):
        return "en"
    else:
        logging.warning(f"Could not determine language for {Path(filename).name}, defaulting to 'en'.")
        return "en"


def process_single_pdf(pdf_path: Path, base_output_dir: Path, export_images: bool):
    """
    Processes a single PDF, converting each page to Markdown, optionally saving page images,
    and saving structured data to a JSONL file and supplementary metadata files.
    """
    if not pdf_path.is_file():
        logging.error(f"Input PDF not found: {pdf_path}")
        return False

    filename = pdf_path.name
    stem = pdf_path.stem

    # Create a directory for this specific PDF's output
    pdf_specific_dir = base_output_dir / stem
    pdf_specific_dir.mkdir(parents=True, exist_ok=True)

    # Create a subdirectory for page images if exporting is enabled
    img_dir = pdf_specific_dir / IMG_SUBDIR
    if export_images:
        img_dir.mkdir(exist_ok=True)

    output_filename = pdf_specific_dir / f"{stem}_pages.jsonl"
    language = get_language_from_filename(filename)

    logging.info(f"Processing: {filename} (Language: {language}, Export Images: {export_images})")

    # Initialize stats collection
    total_pages_saved = 0
    blank_pages = 0
    token_counts = []

    try:
        # Get all page markdowns at once
        markdown_pages_list = pymupdf4llm.to_markdown(str(pdf_path), page_chunks=True)
        num_md_pages = len(markdown_pages_list)
        logging.info(f"Converted {filename} to Markdown, received {num_md_pages} page chunks.")

        # Open PDF with fitz only if we need to export images
        pdf_doc = fitz.open(pdf_path) if export_images else None

        with open(output_filename, 'w', encoding='utf-8') as outfile:
            for page_num_zero_based, page_dict in enumerate(markdown_pages_list):
                page_md = page_dict.get('text', '')

                # Check for content and update stats
                if not page_md.strip():
                    blank_pages += 1
                    continue

                token_counts.append(len(page_md.split()))

                page_data = {
                    "doc_id": filename,
                    "language": language,
                    "page_num": page_num_zero_based + 1,
                    "markdown_content": page_md.strip()
                }

                # Save PNG image if enabled
                if export_images and pdf_doc:
                    try:
                        page_obj = pdf_doc[page_num_zero_based]
                        pix = page_obj.get_pixmap(matrix=fitz.Matrix(IMAGE_SCALE, IMAGE_SCALE))
                        img_name = f"{page_num_zero_based + 1:04}.png"
                        pix.save(img_dir / img_name)
                        # Use a relative POSIX path for cross-platform compatibility
                        page_data["page_img"] = str((Path(IMG_SUBDIR) / img_name).as_posix())
                    except Exception as ie:
                        logging.warning(f"Image export failed for page {page_num_zero_based + 1}: {ie}")

                outfile.write(json.dumps(page_data, ensure_ascii=False) + '\n')
                total_pages_saved += 1

        if pdf_doc:
            pdf_doc.close()

        logging.info(f"Successfully processed and saved content for {total_pages_saved} pages.")
        logging.info(f"Structured page data saved to: {output_filename}")

        # --- After loop – QC & stats file ---
        avg_tok = statistics.mean(token_counts) if token_counts else 0
        total_pages = num_md_pages

        # QC Checks
        blank_ratio_threshold = QC.get('blank_ratio_warn', 0.2)
        if total_pages > 0 and (blank_pages / total_pages) > blank_ratio_threshold:
            logging.warning(
                f"High blank page ratio: {blank_pages}/{total_pages} pages are blank (> {blank_ratio_threshold:.0%}).")

        avg_tokens_threshold = QC.get('avg_tokens_warn', 40)
        if avg_tok < avg_tokens_threshold:
            logging.warning(
                f"Low token density: Average of {avg_tok:.1f} tokens/page is below threshold of {avg_tokens_threshold}.")

        # Write stats file
        stats = {
            "pages_total": total_pages,
            "pages_with_content": len(token_counts),
            "pages_blank": blank_pages,
            "avg_tokens_per_page": round(avg_tok, 1),
            "processed_at_utc": datetime.datetime.utcnow().isoformat() + "Z"
        }
        stats_path = pdf_specific_dir / f"{stem}_parse_stats.json"
        stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
        logging.info(f"Parse statistics saved to: {stats_path}")

        # --- Side-car manual_meta.json ---
        meta = {
            "source_pdf": filename,
            "source_url": None,
            "license_type": None,
            "license_note": "To be filled in manually.",
            "date_accessed": datetime.date.today().isoformat(),
            "num_pages": total_pages
        }
        meta_path = pdf_specific_dir / f"{stem}_manual_meta.json"
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        logging.info(f"Manual metadata template saved to: {meta_path}")

        return True

    except Exception as e:
        logging.error(f"An unexpected error occurred while processing {filename}: {e}", exc_info=True)
        return False


def _output_exists_for(pdf_path: Path, base_output_dir: Path) -> bool:
    stem = pdf_path.stem
    jsonl = base_output_dir / stem / f"{stem}_pages.jsonl"
    return jsonl.exists()


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Process PDFs into per-page Markdown (JSONL) and optional page images.\n\n"
            "Single-file mode:\n"
            "  parse_pdf.py <pdf_filename> [--no-images]\n\n"
            "Bulk mode:\n"
            "  parse_pdf.py --all [--pattern '*.pdf'] [--recurse] [--overwrite] [--no-images]"
        ),
        formatter_class=argparse.RawTextHelpFormatter
    )
    # Modes
    parser.add_argument("pdf_filename", nargs="?", help=f"Filename inside '{PDF_DIR}' to process (single-file mode).")
    parser.add_argument("--all", action="store_true", help="Process all PDFs from the configured pdf_directory.")
    # Discovery options
    parser.add_argument("--pattern", default="*.pdf", help="Glob for selecting PDFs (default: *.pdf).")
    parser.add_argument("--recurse", action="store_true", help="Recurse into subdirectories of pdf_directory.")
    # Behavior toggles
    parser.add_argument("--overwrite", action="store_true", help="Re-process even if output already exists.")
    parser.add_argument("--no-images", action="store_true", help="Disable image export (overrides config).")

    args = parser.parse_args(argv)
    export_images_enabled = EXPORT_IMAGES and not args.no_images

    # ---- Single-file mode
    if args.pdf_filename and not args.all:
        pdf_full_path = PDF_DIR / args.pdf_filename
        logging.info(f"--- Starting PDF processing for: {pdf_full_path} ---")
        ok = process_single_pdf(pdf_full_path, PROCESSED_DATA_DIR, export_images=export_images_enabled)
        if ok:
            # Preview first lines
            stem = pdf_full_path.stem
            output_jsonl = PROCESSED_DATA_DIR / stem / f"{stem}_pages.jsonl"
            try:
                with open(output_jsonl, 'r', encoding='utf-8') as f:
                    print(f"\n--- First 3 lines of output file ({output_jsonl.name}) ---")
                    for i, line in enumerate(f):
                        if i >= 3:
                            break
                        print(line.strip()[:200] + ("..." if len(line) > 200 else ""))
                    print("--------------------------------------------------")
            except FileNotFoundError:
                logging.warning(f"Could not read output file {output_jsonl} for verification.")
            return 0
        else:
            logging.error("--- PDF processing failed. See logs for details. ---")
            return 1

    # ---- Bulk mode
    if args.all:
        if args.recurse:
            candidates = sorted(PDF_DIR.rglob(args.pattern))
        else:
            candidates = sorted(PDF_DIR.glob(args.pattern))

        if not candidates:
            logging.warning(f"No PDFs matched pattern '{args.pattern}' in {PDF_DIR} (recurse={args.recurse}).")
            return 0

        logging.info(
            f"Bulk processing: found {len(candidates)} file(s) in {PDF_DIR} "
            f"(pattern='{args.pattern}', recurse={args.recurse}, images={export_images_enabled})."
        )

        processed = skipped = failed = 0
        for pdf_path in candidates:
            if not args.overwrite and _output_exists_for(pdf_path, PROCESSED_DATA_DIR):
                logging.info(f"Skipping (already processed): {pdf_path.name}")
                skipped += 1
                continue

            logging.info(f"Processing: {pdf_path.name}")
            ok = process_single_pdf(pdf_path, PROCESSED_DATA_DIR, export_images=export_images_enabled)
            if ok:
                processed += 1
            else:
                failed += 1

        logging.info(
            f"Bulk summary → processed: {processed}, skipped: {skipped}, failed: {failed} "
            f"(directory={PDF_DIR}, pattern='{args.pattern}', recurse={args.recurse})"
        )
        # Non-zero exit if any failures occurred (useful in CI)
        return 0 if failed == 0 else 2

    # If we reach here, the user provided neither a filename nor --all
    parser.error("Provide a <pdf_filename> for single-file mode, or use --all for bulk mode.")


if __name__ == "__main__":
    raise SystemExit(main())