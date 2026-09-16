# IndicDLP Local Tar Loader Investigation & Plan

**Date:** 2026-09-16  
**Purpose:** Investigation of prepare_indicdlp.py HuggingFace dependencies and design plan for local tar-based loading

---

## STEP 0 — Housekeeping Check: normalize_category_name() Presence

**Status:** ✅ **YES** - Function is present and committed

**Committed version (commit 836f440):**
```python
def normalize_category_name(name: str) -> str:
    """Normalize category names from JSON annotations to match class_map.yaml format.
    
    IndicDLP annotations use hyphens (e.g., 'chapter-title', 'figure-caption')
    but class_map.yaml uses underscores (e.g., 'chapter_title', 'figure_caption').
    This function bridges the gap without modifying the protected config file.
    """
    return name.strip().lower().replace("-", "_")
```

**Applied at 4 locations:**
1. Line 147: `_iter_class_names()` - dict branch
2. Line 153: `_iter_class_names()` - list branch  
3. Line 166: `_iter_annotations()` - dict branch
4. Line 174: `_iter_annotations()` - list branch

---

## STEP 1 — HuggingFace Hub Dependencies

### Complete list of HF-dependent code:

#### 1. **`load_hf_dataset()` function (lines 66-70)**
```python
def load_hf_dataset(repo: str, split: str, streaming: bool = True):
    from datasets import load_dataset

    log.info("loading %s split=%s streaming=%s", repo, split, streaming)
    return load_dataset(repo, split=split, streaming=streaming)
```
- **Import:** `from datasets import load_dataset`
- **HF API call:** `load_dataset(repo, split=split, streaming=streaming)`
- **Used in:** `inspect()`, `export()`

#### 2. **`inspect()` function (line 84)**
```python
ds = load_hf_dataset(repo, split, streaming=True)
```
- Expects: iterable dataset object from HF Hub
- Iterates with: `for i, row in enumerate(ds):`
- Assumes row structure: `row.keys()`, `row.get("objects")`, etc.

#### 3. **`export()` function (line 218)**
```python
try:
    ds = load_hf_dataset(repo, split, streaming=True)
except Exception as exc:
    log.warning("could not load split %r (%s) -- skipping", split, exc)
    continue
```
- Same pattern as inspect()
- Iterates: `for row in ds:`

#### 4. **Data access pattern throughout `export()` (lines 229-233)**
```python
image = row.get("image")
if image is None:
    continue
img_w, img_h = image.size
```
- Assumes `row["image"]` is a PIL Image object (HF datasets auto-converts)
- Uses `.size` attribute directly

#### 5. **Argparse default (line 325)**
```python
p.add_argument("--repo", default="ai4bharat/indicdlp", help="HF dataset repo id")
```
- Default assumes HF Hub repo ID format

### Summary of HF dependencies:
- **Total functions with HF calls:** 3 (`load_hf_dataset`, `inspect`, `export`)
- **Key assumption:** Dataset rows contain PIL Image objects, not file paths
- **Streaming mode:** Always True (designed to avoid downloading full 91GB)

---

## STEP 2 — Format-Agnostic Components

### Components that do NOT depend on data source:

#### 1. **Category normalization (lines 56-64)**
```python
def normalize_category_name(name: str) -> str:
    """..."""
    return name.strip().lower().replace("-", "_")
```
- **Input:** String category name
- **Output:** Normalized string
- **✓ Fully reusable**

#### 2. **Field name normalization (lines 52-53)**
```python
def _norm(value: str) -> str:
    return str(value).strip().lower().replace(" ", "_").replace("-", "_").replace("&", "and")
```
- **✓ Fully reusable**

#### 3. **Field extraction helper (lines 73-78)**
```python
def _get_field(row: dict, candidates: tuple[str, ...]) -> str | None:
    for key in candidates:
        if key in row and row[key] is not None:
            return str(row[key])
    return None
```
- **Input:** Generic dict
- **✓ Fully reusable**

#### 4. **Category name iteration (lines 137-154)**
```python
def _iter_class_names(row: dict):
    """Yield category names from whichever annotation shape this row uses."""
    # ... (full function)
```
- **Input:** Dict with `objects`/`annotations`/`regions` field
- **✓ Fully reusable** (works with any dict structure)

#### 5. **Annotation extraction (lines 157-175)**
```python
def _iter_annotations(row: dict):
    """Yield (class_name, [x, y, w, h]) pairs in COCO xywh, absolute pixels."""
    # ... (full function)
```
- **Input:** Dict with nested annotation structure
- **Output:** (category_name, bbox) tuples
- **✓ Fully reusable**

#### 6. **YOLO coordinate conversion (lines 178-188)**
```python
def _to_yolo(box_xywh: list[float], img_w: int, img_h: int) -> tuple[float, float, float, float] | None:
    """COCO [x, y, w, h] absolute -> YOLO [cx, cy, w, h] normalised."""
    # ... (full function)
```
- **Input:** Numeric bbox + dimensions
- **✓ Fully reusable**

#### 7. **Class remapping logic (lines 197, 234-247)**
```python
cm = ClassMap.load()
# ...
resolved = cm.resolve(name)
# ...
target, cls_idx = resolved
yolo = _to_yolo(box, img_w, img_h)
lines.append(f"{cls_idx} " + " ".join(f"{v:.6f}" for v in yolo))
stats["by_class"][target] += 1
```
- **✓ Fully reusable** - only needs normalized category names

#### 8. **YOLO file writer (lines 251-252)**
```python
(out_dir / "labels" / split / f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")
```
- **✓ Fully reusable**

#### 9. **Image saver (line 250)**
```python
image.convert("RGB").save(out_dir / "images" / split / f"{stem}.jpg", quality=92)
```
- **Requires:** PIL Image object (but easy to adapt from file)
- **✓ Mostly reusable** (just need to open image from tar first)

#### 10. **Statistics tracking (lines 202-205, 253-258)**
```python
stats = {"seen": 0, "kept": 0, "no_annotations": 0, ...}
# ... various stats["key"] += 1 throughout
```
- **✓ Fully reusable**

#### 11. **Output writers (lines 286-319)**
```python
def _write_data_yaml(out_dir: Path, targets: list[str], splits: tuple[str, ...]) -> None:
    # ...

def _write_stats(out_dir: Path, stats: dict, languages: list[str], domains: list[str]) -> None:
    # ...
```
- **✓ Fully reusable**

#### 12. **Filename stem generation (lines 273-282)**
```python
def _stem(row: dict, split: str, index: int, rng: random.Random) -> str:
    # ...
```
- **✓ Fully reusable** (works with any dict containing filename fields)

### Summary:
**~85% of the code is format-agnostic** - only the data loading and PIL Image access need replacement.

---

## STEP 3 — Filter Logic Field Name Analysis

### Current filter implementation (lines 224-230):

```python
lang = _norm(_get_field(row, ("language", "lang", "language_name")) or "")
dom = _norm(_get_field(row, ("domain", "category", "doc_type", "document_type")) or "")
if lang_filter and lang and lang not in lang_filter:
    continue
if dom_filter and dom and dom not in dom_filter:
    continue
```

### Field names the script looks for:

| Purpose | Candidate Keys (in priority order) | After _norm() |
|---------|-----------------------------------|---------------|
| Language | `"language"`, `"lang"`, `"language_name"` | lowercase, underscored |
| Domain | `"domain"`, `"category"`, `"doc_type"`, `"document_type"` | lowercase, underscored |

### Actual JSON field names (from investigation):

```json
{
  "file_name": "rp_as_000209_0.png",
  "id": 0,
  "language": "Assamese",
  "document_category": "Research papers",
  "width": 1654,
  "height": 2339
}
```

| Field | Actual JSON Key | Actual Values | _norm() Result |
|-------|----------------|---------------|----------------|
| Language | **`"language"`** | `"Assamese"`, `"Hindi"`, `"Marathi"`, etc. | `"assamese"`, `"hindi"`, `"marathi"` |
| Domain | **`"document_category"`** | `"Research papers"`, `"Forms"`, `"Acts_Rules"`, `"Question_paper"`, etc. | `"research_papers"`, `"forms"`, `"acts_rules"`, `"question_paper"` |

### Match Analysis:

✅ **LANGUAGE: EXACT MATCH**
- Script looks for: `"language"` (first candidate)
- JSON has: `"language"`
- **No changes needed**

⚠️ **DOMAIN: PARTIAL MATCH**
- Script looks for: `"domain"`, `"category"`, `"doc_type"`, `"document_type"`
- JSON has: `"document_category"`
- **"document_category" is NOT in the candidate list**
- Current code would get empty string, failing to filter properly

### Required fix:
Add `"document_category"` to the domain candidate tuple:
```python
dom = _get_field(row, ("domain", "category", "doc_type", "document_type", "document_category"))
```

### Filter value normalization:

**Script normalizes both sides:**
1. User input: `DEFAULT_DOMAINS = ["acts_and_rules", "forms", "notices", "newspapers", "question_papers"]`
2. JSON values: `"Acts_Rules"` → `_norm()` → `"acts_rules"` ✓

**Mismatch found:**
- Default: `"acts_and_rules"` (with `and`)
- JSON after _norm: `"acts_rules"` (no `and`)
- `_norm()` replaces `"&"` with `"and"`, but underscore is already present in JSON

**Resolution:** Either:
1. Update `DEFAULT_DOMAINS` to match JSON: `"acts_rules"` not `"acts_and_rules"`
2. Or add pre-normalization alias handling

---

## STEP 4 — Existing Local-File Escape Hatch

### Search results: **NONE FOUND**

**Argparse arguments searched:**
- Line 324: `--repo` (HF repo ID, no local path option)
- Line 325: `--inspect` (boolean flag)
- Line 326: `--inspect-limit` (int)
- Line 327: `--out` (output directory only)
- Line 328-333: `--languages`, `--domains`, `--max-images`, `--splits`, `--seed`, `--on-unmapped`

**No flags for:**
- `--local-path` ❌
- `--from-tar` ❌
- `--offline` ❌
- `--annotations-json` ❌
- `--images-dir` ❌

**Conclusion:** The script is **HF-only with zero local-file support**. Any local tar functionality must be added from scratch.

---

## STEP 5 — Local Tar Loader Design Plan

### Design Principles:
1. **Minimal code changes** - reuse all format-agnostic components
2. **No full extraction** - stream images from tar via `tarfile.extractfile()`
3. **Pre-filtered annotation index** - use already-extracted JSON to build image list
4. **Preserve existing CLI** - add `--from-local-tar` flag, keep everything else unchanged
5. **Same output format** - YOLO structure identical to HF path

### Proposed Implementation Plan:

#### **Function 1: `load_local_annotations(json_path: Path, split: str) -> dict`**
**Purpose:** Load pre-extracted COCO JSON  
**Input:** `d:\AksharDrishti_1\indicdlp_annotations_only\indicdlp\annotations\instances_{split}2017.json`  
**Output:** Dict with `{"images": [...], "annotations": [...], "categories": [...]}`  
**Reuses:** Nothing (new loader)

#### **Function 2: `build_filtered_image_list(annotations_dict: dict, lang_filter: set, dom_filter: set, max_images: int, seed: int) -> list[dict]`**
**Purpose:** Pre-filter images by language/domain before touching tar  
**Steps:**
1. Parse `annotations_dict["images"]` list
2. For each image dict, check `image["language"]` and `image["document_category"]`
3. Apply `_norm()` to both fields
4. Keep only images matching lang_filter AND dom_filter
5. Shuffle with `random.Random(seed)`
6. Return first `max_images` image dicts  
**Reuses:** `_norm()` helper  
**Output:** List of image dicts with fields: `id`, `file_name`, `language`, `document_category`, `width`, `height`

#### **Function 3: `map_annotations_to_images(annotations_dict: dict, filtered_image_ids: set[int]) -> dict[int, list[dict]]`**
**Purpose:** Build lookup: image_id → list of annotations  
**Steps:**
1. Create dict: `ann_map = defaultdict(list)`
2. For each annotation in `annotations_dict["annotations"]`:
   - If `ann["image_id"]` in `filtered_image_ids`:
     - Append annotation to `ann_map[image_id]`
3. Return `ann_map`  
**Reuses:** Nothing (new indexing)  
**Output:** `{image_id: [ann1, ann2, ...], ...}`

#### **Function 4: `map_category_id_to_name(annotations_dict: dict) -> dict[int, str]`**
**Purpose:** Build lookup: category_id → category_name  
**Steps:**
1. Parse `annotations_dict["categories"]`
2. Return `{cat["id"]: cat["name"] for cat in categories}`  
**Reuses:** Nothing (new indexing)  
**Output:** `{0: "advertisement", 1: "answer", ...}`

#### **Function 5: `iter_tar_images(tar_path: Path, split: str, image_list: list[dict]) -> Iterator[tuple[dict, PIL.Image.Image]]`**
**Purpose:** Stream images from tar without full extraction  
**Steps:**
1. `tar = tarfile.open(tar_path, "r")`
2. Build filename → tarinfo mapping: `{member.name: member for member in tar.getmembers() if member.isfile()}`
3. For each image_dict in image_list:
   - Construct tar path: `f"indicdlp/{split}2017/{image_dict['file_name']}"`
   - Get tarinfo from mapping
   - Extract file: `f = tar.extractfile(tarinfo)`
   - Open as PIL: `img = Image.open(f)`
   - Yield `(image_dict, img)`  
**Reuses:** Nothing (new tar streaming)  
**Output:** Iterator of (metadata_dict, PIL_Image) tuples

#### **Function 6: `export_from_local_tar(...)`**
**Purpose:** Replace `export()` with tar-based version  
**Parameters:** Same as current `export()` but replace `repo: str` with `tar_path: Path, annotations_dir: Path`  
**Steps:**
1. Load ClassMap (reuse existing)
2. Create output directories (reuse existing)
3. For each split in splits:
   - Call `load_local_annotations(annotations_dir / f"instances_{split}2017.json", split)`
   - Call `build_filtered_image_list(...)` → get filtered_images
   - Call `map_annotations_to_images(...)` → get ann_map
   - Call `map_category_id_to_name(...)` → get cat_map
   - For each (img_dict, pil_img) from `iter_tar_images(tar_path, split, filtered_images)`:
     - Get annotations: `anns = ann_map[img_dict["id"]]`
     - For each annotation:
       - Get category name: `cat_name = cat_map[ann["category_id"]]`
       - Normalize: `norm_name = normalize_category_name(cat_name)` ← **REUSE**
       - Resolve to target: `resolved = cm.resolve(norm_name)` ← **REUSE**
       - Extract bbox: `bbox = ann["bbox"]` (COCO format)
       - Convert to YOLO: `yolo = _to_yolo(bbox, img_w, img_h)` ← **REUSE**
       - Build YOLO line: `f"{cls_idx} {cx} {cy} {w} {h}"` ← **REUSE**
     - Save image: `pil_img.convert("RGB").save(...)` ← **REUSE**
     - Save labels: `write_text("\n".join(lines))` ← **REUSE**
     - Update stats ← **REUSE**
4. Write data.yaml ← **REUSE `_write_data_yaml()`**
5. Write subset_stats.json ← **REUSE `_write_stats()`**

**Reuses:** 
- `normalize_category_name()`
- `_to_yolo()`
- `ClassMap.resolve()`
- `_write_data_yaml()`
- `_write_stats()`
- `_stem()` for filename generation
- All stats tracking logic

**New code:** ~200 lines for functions 1-5, ~50 lines to wire into function 6

#### **Function 7: `main()` modifications**
**Changes:**
1. Add argparse argument:
   ```python
   p.add_argument("--from-local-tar", type=Path, 
                  help="Use local indicdlp.tar instead of HF Hub (requires --annotations-dir)")
   p.add_argument("--annotations-dir", type=Path,
                  help="Directory containing instances_*2017.json files")
   ```
2. Add conditional dispatch:
   ```python
   if args.from_local_tar:
       if not args.annotations_dir:
           p.error("--from-local-tar requires --annotations-dir")
       export_from_local_tar(
           tar_path=args.from_local_tar,
           annotations_dir=args.annotations_dir,
           out_dir=args.out,
           languages=args.languages,
           domains=args.domains,
           max_images=args.max_images,
           splits=tuple(args.splits),
           seed=args.seed,
           on_unmapped=args.on_unmapped
       )
   else:
       export(...)  # existing HF path
   ```

#### **Required fixes for existing code:**
1. Add `"document_category"` to domain field candidates in `export()` line 226
2. Update `DEFAULT_DOMAINS` line 48 to match actual JSON values:
   ```python
   DEFAULT_DOMAINS = ["acts_rules", "forms", "notice", "newspaper", "question_paper"]
   ```
   (Note: plural/singular mismatches need verification too)

### Usage example:
```bash
python data/prepare_indicdlp.py \
  --from-local-tar d:/AksharDrishti_1/indicdlp.tar \
  --annotations-dir d:/AksharDrishti_1/indicdlp_annotations_only/indicdlp/annotations \
  --out datasets/indicdlp_subset \
  --languages hindi marathi \
  --domains forms acts_rules notice newspaper question_paper \
  --max-images 15000
```

### Estimated code size:
- **New functions 1-5:** ~150-200 lines
- **Modified `main()`:** +20 lines
- **New `export_from_local_tar()`:** ~150 lines (mostly copy-paste from `export()` with tar iteration)
- **Bug fixes:** ~5 lines
- **Total new/modified:** ~350 lines

### Benefits:
- ✅ No HF Hub authentication needed
- ✅ No 91GB download
- ✅ Streams from tar (minimal disk usage)
- ✅ Reuses 85% of existing code
- ✅ Same output format (drop-in replacement)
- ✅ Preserves HF path for users who have access

---

## Summary

### STEP 0: ✅ normalize_category_name() confirmed present in commit 836f440

### STEP 1: 5 HF-dependent code blocks identified
- `load_hf_dataset()`, `inspect()`, `export()` - all use `datasets.load_dataset()`
- Row iteration assumes HF Dataset object with PIL Image in `row["image"]`

### STEP 2: ~85% of code is format-agnostic
- All class remapping, YOLO conversion, file writing fully reusable
- Only data loading and image access need replacement

### STEP 3: Field name match analysis
- **Language:** ✅ Exact match (`"language"`)
- **Domain:** ⚠️ **Mismatch** - JSON has `"document_category"`, script looks for `"domain"`/`"category"`/`"doc_type"`/`"document_type"` (missing from candidate list)
- **Value normalization mismatch:** `"acts_and_rules"` in defaults vs `"acts_rules"` in JSON

### STEP 4: No local-file support exists - must build from scratch

### STEP 5: 7-function design plan with ~350 lines of new/modified code
- Functions 1-5: Annotation loading, filtering, indexing
- Function 6: Tar-streaming export (reuses 85% of existing logic)
- Function 7: CLI integration with `--from-local-tar` flag
