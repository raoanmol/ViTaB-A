"""Shared utilities for table format conversion, markdown rendering, and image generation."""

import base64


def simplify_table(hitab_json: dict) -> dict:
    """Convert a HiTab-style table dict to a simplified format.

    Resolves merged regions, splits header from data rows,
    and drops internal metadata fields.
    """
    # Deep copy the texts grid (don't mutate the input)
    texts = [row[:] for row in hitab_json["texts"]]

    # Resolve merged regions
    for region in hitab_json.get("merged_regions", []):
        anchor_value = texts[region["first_row"]][region["first_column"]]
        for r in range(region["first_row"], region["last_row"] + 1):
            if r >= len(texts):
                continue
            for c in range(region["first_column"], region["last_column"] + 1):
                if c >= len(texts[r]):
                    continue
                texts[r][c] = anchor_value

    # Split into header and rows
    top_n = hitab_json.get("top_header_rows_num", 0)
    header = texts[:top_n] if top_n > 0 else []
    rows = texts[top_n:]

    return {
        "title": hitab_json.get("title", ""),
        "header": header,
        "rows": rows,
    }


def column_letter(col_idx: int) -> str:
    """Convert 0-indexed column number to Excel-style letter (A, B, ..., Z, AA, AB, ...)."""
    letter = ''
    col_idx += 1
    while col_idx > 0:
        col_idx -= 1
        letter = chr(col_idx % 26 + ord('A')) + letter
        col_idx //= 26
    return letter


def table_to_markdown(hitab_json: dict) -> str:
    """Convert HiTab-style table JSON to a markdown table with row/column labels."""
    texts = hitab_json['texts']
    merged_regions = hitab_json.get('merged_regions', [])
    title = hitab_json.get('title', '')

    num_rows = len(texts)
    num_cols = len(texts[0]) if texts else 0

    # Track which cells to skip (non-anchor cells of merged regions)
    skip_cells = set()
    for region in merged_regions:
        first_row, last_row = region['first_row'], region['last_row']
        first_col, last_col = region['first_column'], region['last_column']
        for r in range(first_row, last_row + 1):
            for c in range(first_col, last_col + 1):
                if r != first_row or c != first_col:
                    skip_cells.add((r, c))

    markdown = ""
    current_row = 1

    # Column header row: | 1 | A | B | C | ...
    col_labels = [str(current_row)] + [column_letter(i) for i in range(num_cols)]
    markdown += "| " + " | ".join(col_labels) + " |\n"
    current_row += 1

    # Separator row (no row number)
    markdown += "| " + " | ".join(["---"] * (num_cols + 1)) + " |\n"

    # Title row if present
    if title:
        title_row = [str(current_row), title] + [""] * (num_cols - 1)
        markdown += "| " + " | ".join(title_row) + " |\n"
        current_row += 1

    # Data rows
    for row_idx in range(num_rows):
        row_cells = [str(current_row)]
        current_row += 1

        for col_idx in range(num_cols):
            if (row_idx, col_idx) in skip_cells:
                cell_value = ""
            else:
                cell_value = texts[row_idx][col_idx]
            row_cells.append(str(cell_value))

        markdown += "| " + " | ".join(row_cells) + " |\n"

    return markdown


def table_to_html(hitab_json: dict, font_family: str = 'Arial',
                  header_bg_color: str = '#f0f0f0',
                  header_text_color: str = '#000000') -> str:
    """Convert HiTab-style table JSON to styled HTML for screenshot rendering."""
    texts = hitab_json['texts']
    merged_regions = hitab_json.get('merged_regions', [])
    top_header_rows = hitab_json.get('top_header_rows_num', 0)
    title = hitab_json.get('title', '')
    if title:
        top_header_rows = max(0, top_header_rows - 1)

    num_rows = len(texts)
    num_cols = len(texts[0]) if texts else 0

    # Build merge lookup and skip set
    merged_cells = {}
    skip_cells = set()
    for region in merged_regions:
        first_row, last_row = region['first_row'], region['last_row']
        first_col, last_col = region['first_column'], region['last_column']
        rowspan = last_row - first_row + 1
        colspan = last_col - first_col + 1
        merged_cells[(first_row, first_col)] = (rowspan, colspan)
        for r in range(first_row, last_row + 1):
            for c in range(first_col, last_col + 1):
                if r != first_row or c != first_col:
                    skip_cells.add((r, c))

    html = f"""
    <html>
    <head>
        <style>
            body {{
                font-family: {font_family}, sans-serif;
                padding: 20px;
                background-color: white;
            }}
            table {{
                border-collapse: collapse;
                margin: 0 auto;
                background-color: white;
                font-family: {font_family}, sans-serif;
            }}
            td, th {{
                border: 1px solid #333;
                padding: 8px 12px;
                text-align: center;
                font-size: 14px;
                font-family: {font_family}, sans-serif;
            }}
            .header-cell {{
                background-color: {header_bg_color};
                color: {header_text_color};
                font-weight: bold;
            }}
            .row-label, .col-label {{
                background-color: #e0e0e0;
                font-weight: bold;
                font-size: 12px;
                color: #333;
            }}
            .table-title {{
                font-size: 16px;
                font-weight: bold;
                text-align: center;
                font-family: {font_family}, sans-serif;
            }}
        </style>
    </head>
    <body>
        <table>
    """

    current_row = 1

    # Column labels row
    html += "        <tr>\n"
    html += f'            <td class="row-label">{current_row}</td>\n'
    for col_idx in range(num_cols):
        html += f'            <td class="col-label">{column_letter(col_idx)}</td>\n'
    html += "        </tr>\n"
    current_row += 1

    # Title row if present
    if title:
        html += "        <tr>\n"
        html += f'            <td class="row-label">{current_row}</td>\n'
        html += f'            <td class="table-title" colspan="{num_cols}">{title}</td>\n'
        html += "        </tr>\n"
        current_row += 1

    # Data rows
    for row_idx in range(num_rows):
        html += "        <tr>\n"
        html += f'            <td class="row-label">{current_row}</td>\n'
        current_row += 1

        for col_idx in range(num_cols):
            if (row_idx, col_idx) in skip_cells:
                continue

            cell_value = texts[row_idx][col_idx]
            is_header = row_idx < top_header_rows
            cell_class = 'header-cell' if is_header else ''

            rowspan, colspan = merged_cells.get((row_idx, col_idx), (1, 1))

            span_attrs = ""
            if rowspan > 1:
                span_attrs += f' rowspan="{rowspan}"'
            if colspan > 1:
                span_attrs += f' colspan="{colspan}"'

            html += f'            <td class="{cell_class}"{span_attrs}>{cell_value}</td>\n'

        html += "        </tr>\n"

    html += """
        </table>
    </body>
    </html>
    """

    return html


VARIATIONS = {
    'arial': {
        'font_family': 'Arial',
        'header_bg_color': '#f0f0f0',
        'header_text_color': '#000000',
    },
    'times_new_roman': {
        'font_family': 'Times New Roman',
        'header_bg_color': '#f0f0f0',
        'header_text_color': '#000000',
    },
    'red': {
        'font_family': 'Arial',
        'header_bg_color': '#ff6b6b',
        'header_text_color': '#ffffff',
    },
    'blue': {
        'font_family': 'Arial',
        'header_bg_color': '#4a90e2',
        'header_text_color': '#ffffff',
    },
    'green': {
        'font_family': 'Arial',
        'header_bg_color': '#51cf66',
        'header_text_color': '#ffffff',
    },
}


def render_html_to_base64(page, html_content: str) -> str:
    """Render HTML to a base64-encoded PNG screenshot using a Playwright page.

    Args:
        page: A playwright.sync_api.Page object (caller manages browser lifecycle).
        html_content: Full HTML document string to render.

    Returns:
        Base64-encoded PNG string.
    """
    page.set_content(html_content)
    page.wait_for_load_state('networkidle')
    screenshot_bytes = page.screenshot(type='png', full_page=True)
    return base64.b64encode(screenshot_bytes).decode('utf-8')


def generate_table_images(hitab_json: dict, page, variants_to_render: list[str]) -> dict:
    """Generate styled table images for specified variants.

    Args:
        hitab_json: HiTab-style table dict with texts, merged_regions, etc.
        page: A playwright.sync_api.Page object.
        variants_to_render: List of variant names to actually render (e.g. ["arial", "red"]).
            Must be keys from VARIATIONS.

    Returns:
        Dict with all 5 variant keys. Rendered variants have base64 PNG strings,
        non-rendered variants have empty string "".
    """
    images = {name: "" for name in VARIATIONS}

    for variant_name in variants_to_render:
        style = VARIATIONS[variant_name]
        html = table_to_html(
            hitab_json,
            font_family=style['font_family'],
            header_bg_color=style['header_bg_color'],
            header_text_color=style['header_text_color'],
        )
        images[variant_name] = render_html_to_base64(page, html)

    return images
