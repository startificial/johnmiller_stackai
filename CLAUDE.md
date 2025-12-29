# Claude Code Rules

## Test Documents

The `test_documents/` folder contains PDF files designed to test various text extraction scenarios:

| File | Purpose |
|------|---------|
| `01_happy_path_text_only.pdf` | Basic text extraction test. Contains plain text with paragraphs, numbers (1234567890), and special characters (!@#$%^&*). No tables, images, or complex formatting. Use to verify baseline extraction works correctly. |
| `02_no_content.pdf` | Empty/blank PDF edge case. Tests handling when no extractable content exists. Should return empty result or appropriate error handling. |
| `03_hierarchical_structure.pdf` | Document structure extraction test. Contains multi-level headers (1., 1.1, 1.1.1), header/footer regions, and nested content. Tests preservation of document hierarchy and section relationships. |
| `04_charts.pdf` | Visual element handling test. Contains bar charts, pie charts, and line charts with labels. Tests extraction of chart titles and axis labels, and handling of non-text graphical elements. |
| `05_tables.pdf` | Table extraction test. Contains simple data tables, complex tables with merged/nested cells, and numeric data with currency/percentages. Tests table structure preservation and cell data accuracy. |
| `06_text_cleaning_test.pdf` | Text normalization stress test. Contains problematic patterns: extra whitespace, special characters (em-dashes, bullets, fractions), various number/date/time formats, URLs, emails, dense legal text, and repeated separators (----, ====). Tests text cleaning and edge case handling. |
| `07_multipage_no_hierarchy.pdf` | Multi-page extraction without structure. 28-page financial report (Apex Industries) with flowing paragraphs and embedded tables but NO hierarchical markers. Tests handling of long documents without organizational structure. |
| `08_multipage_with_hierarchy.pdf` | Multi-page extraction with structure. 27-page financial report (Meridian Holdings) WITH clear hierarchy: Table of Contents, numbered sections (1.1.1), headers/footers on each page. Tests comprehensive extraction preserving document organization. |

## Manual QA Tests

When generating manual QA test plans, always save them to the `manual-qa/` folder (not the project root). Use the naming convention: `manual-qa-tests-[timestamp].md`

Example: `manual-qa/manual-qa-tests-20251229_102700.md`

## Ignored Folders

The following folders are not tracked in version control:
- `manual-qa/` - Generated manual QA test plans
