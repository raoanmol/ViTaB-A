import json
import base64
from io import BytesIO
from typing import Optional
from PIL import Image
from src.data.dataset import ViTaBSample

SYSTEM_INSTRUCTION = (
    "You are a table question-answering assistant. Given a question, its answer, "
    "and a table, identify the specific table cell(s) that support the answer. "
    "Return the cell references in Excel-style notation (e.g., =E7, =B3). "
    "Return ONLY the cell references, one per line, with no other text."
)


def get_table_content(sample: ViTaBSample, table_repr: str) -> tuple[str, Optional[Image.Image]]:
    if table_repr == "json":
        return (json.dumps(sample.table_json, indent=2), None)
    elif table_repr == "markdown":
        return (sample.table_md, None)
    elif table_repr.startswith("image_"):
        variant = table_repr[len("image_"):]
        b64 = sample.table_images[variant]
        if not b64:
            raise ValueError(f"Image variant {variant!r} is empty (not rendered)")
        image_bytes = base64.b64decode(b64)
        image = Image.open(BytesIO(image_bytes))
        return ("", image)
    else:
        raise ValueError(f"Unknown table_repr: {table_repr!r}")


def build_prompt(
    sample: ViTaBSample,
    table_repr: str,
    strategy: str = "zero_shot",
) -> tuple[str, Optional[Image.Image]]:
    table_text, image = get_table_content(sample, table_repr)
    answer = str(sample.answer)

    if image is not None:
        prompt_text = (
            f"{SYSTEM_INSTRUCTION}\n\n"
            "The table is shown in the image above.\n\n"
            f"Question: {sample.question}\n"
            f"Answer: {answer}\n\n"
            "Which cell(s) in the table support this answer? Provide Excel-style references (e.g., =E7)."
        )
    else:
        fmt_label = "JSON" if table_repr == "json" else "Markdown"
        prompt_text = (
            f"{SYSTEM_INSTRUCTION}\n\n"
            f"Table ({fmt_label}):\n"
            f"{table_text}\n\n"
            f"Question: {sample.question}\n"
            f"Answer: {answer}\n\n"
            "Which cell(s) in the table support this answer? Provide Excel-style references (e.g., =E7)."
        )

    return (prompt_text, image)
