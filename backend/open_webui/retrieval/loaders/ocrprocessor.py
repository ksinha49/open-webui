"""
Modification Log:
------------------
| Date       | Author         | MOD TAG            | Description                                              |
|------------|----------------|--------------------|----------------------------------------------------------|
| 2024-11-05 | AAK7S          | AMER-ENH           | OCR ENHANCEMENT                                          |
| 2025-02-18 | AAK7S          | AMER-ENH2           | Retain scanned image format and page number              |
|            |                |                    | Added async wrappers for OCR functions                   |
| 2025-04-05 | AAK7S          | AMER-ENH3           | Integrated layout-aware OCR processing into OCRProcessor   |
| 2025-04-06 | AAK7S          | CONF-ADD           | Added confidence score to metadata and preserved line breaks in OCR text |
"""

import pymupdf as fitz  # PyMuPDF
import easyocr  # EasyOCR for fallback OCR extraction
import torch
import multiprocessing  # For parallel processing
from PIL import Image
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor
import gc
import numpy as np  # For image processing
import psutil  # For monitoring system memory
from functools import lru_cache
import json
import cv2  # For image preprocessing
import hashlib
import tempfile
import os
import logging
import time  # For timing batch processing
import asyncio  # ASYNC: AMER-ENH2 For async wrappers

from langchain_core.documents import Document
from open_webui.env import DPI, BATCH_SIZE, ENV_TMP_DIR, SRC_LOG_LEVELS
log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS["RAG"])

image_ext = ["jpg", "jpeg", "png", "tiff", "bmp", "gif"]

use_gpu = torch.cuda.is_available()
log.info(f"GPU available: {use_gpu}")
reader = None
try:
    reader = easyocr.Reader(
        ['en'], gpu=use_gpu, model_storage_directory='models/',
        user_network_directory='user_network', download_enabled=True,
        detector=True, recognizer=True
    )
    log.info("Initialized EasyOCR Reader with optimized settings.")
except RuntimeError as e:
    if "CUDA out of memory" in str(e):
        log.warning("CUDA memory insufficient for OCR. Switching to CPU mode.")
        reader = easyocr.Reader(['en'], gpu=False)
    else:
        log.error(f"Failed to initialize EasyOCR Reader: {str(e)}")
        raise e

# ENH_START: AMER-ENH - Added missing get_checkpoint_path function and ensure ENV_TMP_DIR exists
def get_checkpoint_path(file_path):
    if not os.path.exists(ENV_TMP_DIR):
        try:
            os.makedirs(ENV_TMP_DIR)
        except Exception as e:
            log.error(f"Failed to create ENV_TMP_DIR: {ENV_TMP_DIR}, error: {str(e)}")
            raise
    abs_path = os.path.abspath(file_path)
    hash_digest = hashlib.md5(abs_path.encode('utf-8')).hexdigest()
    checkpoint_filename = f"{hash_digest}_checkpoint.json"
    checkpoint_path = os.path.join(ENV_TMP_DIR, checkpoint_filename)
    log.info(f"Checkpt loc: {checkpoint_path}")
    return checkpoint_path
# ENH_END: AMER-ENH2

# ENH_START: AMER-ENH2 - Modified to return a tuple (page number, image bytes in PNG format)
def extract_image_from_page(page_num, pdf_document, dpi=DPI):
    """Return (page_num, image_bytes) for the specified page."""
    try:
        page = pdf_document.load_page(page_num)
        matrix = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=matrix)
        image_bytes = pix.tobytes("png")
        return (page_num, image_bytes)
    except Exception as e:
        log.error(f"Error extracting image from page {page_num}: {str(e)}")
        return (page_num, None)
# ENH_END: AMER-ENH2

# ENH_START: AMER-ENH2 - Adjusted to collect (page number, image bytes) tuples
def extract_images_from_pages(pdf_document, page_nums, dpi=DPI):
    images = []
    for page_num in page_nums:
        result = extract_image_from_page(page_num, pdf_document, dpi)
        if result and result[1] is not None:
            images.append(result)
    return images
# ENH_END: AMER-ENH2

def preprocess_image_cv2(img_bytes):
    try:
        np_img = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        if img is None:
            log.error("Failed to decode image bytes.")
            return None
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        denoised = cv2.medianBlur(thresh, 3)
        return denoised
    except Exception as e:
        log.error(f"Error preprocessing image: {str(e)}")
        return None

@lru_cache(maxsize=128)
def get_pdf_document(pdf_path):
    try:
        return fitz.open(pdf_path)
    except Exception as e:
        log.error(f"Error opening PDF {pdf_path}: {str(e)}")
        return None

def load_checkpoint(checkpoint_path):
    try:
        with open(checkpoint_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return []
    except json.JSONDecodeError as e:
        log.error(f"Error decoding checkpoint file: {str(e)}")
        return []
    except Exception as e:
        log.error(f"Error loading checkpoint: {str(e)}")
        return []

def save_checkpoint(checkpoint_path, processed_pages):
    try:
        with open(checkpoint_path, 'w') as f:
            json.dump(processed_pages, f)
    except Exception as e:
        log.error(f"Error saving checkpoint: {str(e)}")

class OCRProcessor:
    def __init__(self, ocr_reader, ocr_engine="easyocr"):
        self.ocr_reader = ocr_reader if ocr_reader else reader
        self.ocr_engine = ocr_engine

        # If using an EasyOCR model that contains RNNs, flatten their parameters.
        if hasattr(self.ocr_reader, "model"):
            self.flatten_rnn_parameters(self.ocr_reader.model)

    def flatten_rnn_parameters(self, model):
        """
        Iterate through the model's sub-modules and call flatten_parameters()
        on any LSTM modules to ensure their weights are contiguous.
        """
        for module in model.modules():
            if isinstance(module, torch.nn.LSTM):
                module.flatten_parameters()
                log.info(f"Called flatten_parameters on {module}")

    def perform_ocr(self, img_input):
        try:
            if isinstance(self.ocr_reader, easyocr.Reader):
                # Get detailed OCR results (including bounding box and confidence)
                ocr_results = self.ocr_reader.readtext(img_input)
                # Use layout-aware processing to convert the OCR results into Markdown format
                text, avg_conf = layout_to_markdown(ocr_results)
                return text, avg_conf
            elif self.ocr_engine == "pytesseract":
                img = Image.open(BytesIO(img_input)).convert("RGB")
                text = self.ocr_reader.image_to_string(img)
                return text, None
            elif self.ocr_engine == "paddleocr":
                ocr_results = self.ocr_reader.ocr(img_input, rec=True, cls=True)
                text, avg_conf = layout_to_markdown(ocr_results)
                return text, avg_conf
            else:
                log.error(f"Unsupported OCR engine: {self.ocr_engine}")
                return "", None
        except Exception as e:
            log.error(f"OCR failed: {str(e)}")
            return "", None

def clear_gpu_memory():
    if use_gpu:
        torch.cuda.empty_cache()
    gc.collect()
    if use_gpu:
        allocated = torch.cuda.memory_allocated() / 1e6
        reserved = torch.cuda.memory_reserved() / 1e6
        log.info(f"GPU memory allocated: {allocated:.2f} MB")
        log.info(f"GPU memory reserved: {reserved:.2f} MB")

# ENH_START: AMER-ENH2- Cache OCRProcessor instances to avoid repeated initialization
_ocr_processor_cache = {}

def get_cached_ocr_processor(ocr_reader, ocr_engine):
    key = (id(ocr_reader), ocr_engine)
    if key not in _ocr_processor_cache:
        _ocr_processor_cache[key] = OCRProcessor(ocr_reader, ocr_engine)
    return _ocr_processor_cache[key]
# ENH_END: AMER-ENH2

# ENH_START: AMER-ENH2 - Updated _perform_ocr to use cached OCRProcessor and return confidence score
def _perform_ocr(ocr_reader, ocr_engine, img_bytes):
    ocr_processor = get_cached_ocr_processor(ocr_reader, ocr_engine)
    text, avg_conf = ocr_processor.perform_ocr(img_bytes)
    torch.cuda.empty_cache()
    gc.collect()
    return text, avg_conf
# ENH_END: AMER-ENH2

# NEW: Convert OCR text to Markdown format with page number and original PDF image.
def convert_to_markdown(text: str, page_number=None, img_bytes=None) -> str:
    """
    Convert plain OCR text to Markdown format with page number and optional inline image.
    
    If a page number is provided, it adds a header with the page number (converted to 1-indexed).
    If image bytes are provided, it embeds the image as a base64 encoded inline image.
    The OCR text is then added as plain paragraphs.
    """
    markdown_str = ""
    if page_number is not None:
        # Convert zero-indexed page number to human-readable 1-indexed page number.
        markdown_str += f"\n\n# [Page {page_number + 1}]:\n\n"
    else:
        markdown_str += "# Image Scan Text\n\n"

    # Add the OCR text as plain paragraphs
    markdown_str += text
    return markdown_str

# AMER-ENH3: Post-process OCR text to clean up extra whitespace while preserving line breaks.
def post_process_text(text):
    """
    Post-process OCR output to normalize the text while preserving line breaks.
    This function trims excess whitespace from each line and retains newline separation.
    """
    lines = text.splitlines()
    # Remove extra whitespace from each line and filter out empty lines
    processed_lines = [line.strip() for line in lines if line.strip()]
    # Reassemble the lines using newline characters
    return "\n".join(processed_lines)

# NEW: Helper function to convert OCR results into Markdown while preserving layout.
def layout_to_markdown(ocr_results, vertical_threshold=10):
    """
    Convert OCR results (each a tuple of bounding box, text, confidence)
    into a Markdown string preserving the layout and compute the average confidence.
    
    This heuristic sorts the results by the top y-coordinate and groups text
    into paragraphs if the gap between the bottom of one block and the top of
    the next exceeds vertical_threshold pixels.
    """
    # Sort results by the top y-coordinate of their bounding box.
    sorted_results = sorted(ocr_results, key=lambda res: res[0][0][1])
    
    paragraphs = []
    current_paragraph = []
    last_bottom = None
    confidences = []
    
    for bbox, text, conf in sorted_results:
        confidences.append(conf)
        top = bbox[0][1]  # Top y-coordinate
        if last_bottom is not None and (top - last_bottom) > vertical_threshold:
            paragraphs.append(" ".join(current_paragraph))
            current_paragraph = []
        current_paragraph.append(text)
        # Use the bottom y-coordinate (assuming bbox[2] is bottom-right)
        last_bottom = bbox[2][1]
    
    if current_paragraph:
        paragraphs.append(" ".join(current_paragraph))
    
    # Join paragraphs with double newlines (Markdown paragraph separator)
    markdown_text = "\n\n".join(paragraphs)
    avg_conf = sum(confidences) / len(confidences) if confidences else None
    return markdown_text, avg_conf

# ENH_START: AMER-ENH2 - Modified to use (page number, image bytes) and include metadata with timing logging
def ocr_pdf_fallback(pdf_path, ocr_reader, ocr_engine="easyocr", batch_size=BATCH_SIZE, dpi=DPI):
    extracted_docs = []  # List of Document objects with metadata
    failed_batches = []
    checkpoint_path = get_checkpoint_path(pdf_path)
    processed_pages = load_checkpoint(checkpoint_path)
    try:
        with fitz.open(pdf_path) as pdf_document:
            num_pages = len(pdf_document)
            log.info(f"Starting OCR fallback for {num_pages} pages.")
            for batch_start in range(0, num_pages, batch_size):
                batch_time_start = time.time()  # Start timing the batch
                batch_end = min(batch_start + batch_size, num_pages)
                page_nums = list(range(batch_start, batch_end))
                page_nums = [pn for pn in page_nums if pn not in processed_pages]
                if not page_nums:
                    continue
                log.debug(f"Processing batch: {page_nums}")
                images = extract_images_from_pages(pdf_document, page_nums, dpi)
                if images:
                    try:
                        with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as exec_ocr:
                            future_to_page = {
                                exec_ocr.submit(_perform_ocr, ocr_reader, ocr_engine, img_bytes): (page_num, img_bytes)
                                for page_num, img_bytes in images
                            }
                            for future in future_to_page:
                                page_num, img_bytes = future_to_page[future]
                                if img_bytes is None:
                                    continue
                                text, avg_conf = future.result()
                                if text:
                                    text = post_process_text(text)
                                    text = convert_to_markdown(text, page_num, img_bytes)
                                    doc_metadata = {
                                        "page_number": page_num,
                                        "image_format": "png",
                                        "average_confidence": avg_conf,
                                    }
                                    extracted_docs.append(Document(page_content=text, metadata=doc_metadata))
                                    processed_pages.append(page_num)
                                    save_checkpoint(checkpoint_path, processed_pages)
                                del img_bytes
                                gc.collect()
                                if use_gpu:
                                    torch.cuda.empty_cache()
                    except RuntimeError as e:
                        if "CUDA out of memory" in str(e):
                            log.warning("Reducing batch size due to CUDA out of memory.")
                            batch_size = max(1, batch_size // 2)
                            torch.cuda.empty_cache()
                            continue
                        else:
                            log.error(f"OCR failed for batch {page_nums}: {str(e)}")
                            failed_batches.extend(page_nums)
                    mem = psutil.virtual_memory()
                    if mem.percent > 80:
                        log.warning("High memory usage detected. Reducing batch size.")
                        batch_size = max(1, batch_size // 2)
                        torch.cuda.empty_cache()
                if use_gpu:
                    torch.cuda.empty_cache()
                log.debug(f"Batch {page_nums} processed in {time.time() - batch_time_start:.2f} seconds")
    except Exception as e:
        log.error(f"OCR extraction failed: {str(e)}")
        return []
    if failed_batches:
        log.warning(f"OCR failed for batches: {failed_batches}")
    else:
        try:
            os.remove(checkpoint_path)
            log.debug(f"Deleted checkpoint file: {checkpoint_path}")
        except OSError as e:
            log.error(f"Error deleting checkpoint file {checkpoint_path}: {str(e)}")
    clear_gpu_memory()
    return extracted_docs
# ENH_END: AMER-ENH2

# ENH_START: AMER-ENH2 - Enhanced to preserve image format metadata for scanned image files and use LANCZOS
def ocr_image(image_path, ocr_reader, ocr_engine="easyocr"):
    try:
        with Image.open(image_path) as img:
            original_format = img.format if img.format else "unknown"
            img = img.convert("RGB")
            max_size = (2000, 2000)
            img.thumbnail(max_size, Image.LANCZOS)
            buffer = BytesIO()
            # Save image as PNG to standardize the OCR input while retaining original format info
            img.save(buffer, format="PNG")
            img_bytes = buffer.getvalue()
            preprocessed_img = preprocess_image_cv2(img_bytes)
            if preprocessed_img is None:
                return []
            text, avg_conf = _perform_ocr(ocr_reader, ocr_engine, preprocessed_img.tobytes())
            # Post-process the OCR text.
            text = post_process_text(text)
            # Convert text to Markdown format. For images, no page number is provided.
            text = convert_to_markdown(text, None, img_bytes)
            metadata = {"image_format": original_format, "average_confidence": avg_conf}
            return [Document(page_content=text, metadata=metadata)]
    except Exception as e:
        log.error(f"OCR failed for image {image_path}: {str(e)}")
        return []
# ENH_END: AMER-ENH2

# ASYNC_START: AMER-ENH2 - Added async wrappers for OCR functions
async def async_ocr_pdf_fallback(pdf_path, ocr_reader, ocr_engine="easyocr", batch_size=BATCH_SIZE, dpi=DPI):
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, ocr_pdf_fallback, pdf_path, ocr_reader, ocr_engine, batch_size, dpi)
    return result

async def async_ocr_image(image_path, ocr_reader, ocr_engine="easyocr"):
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, ocr_image, image_path, ocr_reader, ocr_engine)
    return result
# ASYNC_END: AMER-ENH2
