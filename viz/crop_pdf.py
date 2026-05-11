import pymupdf  # Install with: pip install pymupdf


def crop_pdf_to_content(input_pdf, output_pdf, padding=5):
    """
    Automatically crops a PDF to the tightest bounding box of its content.
    """
    doc = pymupdf.open(input_pdf)

    for page in doc:
        content_rect = None

        # Union bboxes from vector drawings
        for drawing in page.get_drawings():
            r = pymupdf.Rect(drawing["rect"])
            content_rect = r if content_rect is None else content_rect | r

        # Union bboxes from images
        for img in page.get_image_info():
            r = pymupdf.Rect(img["bbox"])
            content_rect = r if content_rect is None else content_rect | r

        # Union bboxes from text blocks
        for block in page.get_text("blocks"):
            r = pymupdf.Rect(block[:4])
            content_rect = r if content_rect is None else content_rect | r

        if content_rect and not content_rect.is_empty:
            # Add a small padding (in points) around the content
            content_rect.x0 -= padding
            content_rect.y0 -= padding
            content_rect.x1 += padding
            content_rect.y1 += padding

            # Clamp to page mediabox so cropbox stays valid
            content_rect &= page.mediabox

            # Set the page's CropBox to this new rectangle
            page.set_cropbox(content_rect)

    doc.save(output_pdf)
    doc.close()
    print(f"Successfully cropped: {output_pdf}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Crop a PDF to its content bounding box."
    )
    parser.add_argument("input_pdf", help="Path to the input PDF file.")
    parser.add_argument(
        "output_pdf",
        nargs="?",
        help="Path for the output PDF. Defaults to overwriting the input.",
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=5,
        help="Padding in points around content (default: 5).",
    )
    args = parser.parse_args()

    output = args.output_pdf if args.output_pdf else args.input_pdf
    crop_pdf_to_content(args.input_pdf, output, padding=args.padding)
