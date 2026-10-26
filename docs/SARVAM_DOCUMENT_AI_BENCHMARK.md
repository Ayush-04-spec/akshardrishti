# Sarvam Document AI benchmark

This is an opt-in, hosted OCR benchmark. It does not change the default CRNN
pipeline. The runner uses the approved mapping
`benchmark/mappings/benchmark_image_gt_mapping_v1.csv`, excludes unresolved
BM_018, and expects exactly 23 pages. Input pages are submitted in stable
filename order as flat ZIPs of at most 10 images, using `mr-IN` and requesting
JSON results.

## Preflight and execution

From the application directory, run the offline preflight first:

```powershell
$env:PYTHONUTF8 = '1'
python eval/run_sarvam_benchmark.py --preflight
```

At implementation time, `sarvamai` was not installed and `SARVAM_API_KEY` was
not set. After approval to install the optional SDK and submit these benchmark
documents, install it with:

```powershell
python -m pip install sarvamai
```

Set `SARVAM_API_KEY` privately in the process environment, then run:

```powershell
$env:PYTHONUTF8 = '1'
python eval/run_sarvam_benchmark.py --submit --confirm-upload-and-billing
```

The last command uploads the 23 benchmark pages and may incur charges. The
runner refuses an existing output directory at
`results/sarvam_document_ai_bm23_v1`. It saves each batch's input ZIP and SDK
job, status, and JSON-results response snapshots under `raw/`; normalized
predictions, mapping, page/corpus metrics, runtime, estimated cost, and
failures are saved at the output root. This runner uses the SDK's structured
`get_results(format="json")` response rather than a presigned output download.

## API facts and limitations

- The official Python SDK is `sarvamai`; Document AI uses `SarvamAI.doc_ai` and
  asynchronous `digitise`, `get_status`, and `get_results` calls. The SDK also
  exposes `get_download_url` when a downloaded output bundle is desired.
- With installed `sarvamai==0.1.39`, `digitise()` does not accept
  `output_format`. The runner uses only the installed signature's `file` and
  `language` parameters, polls `get_status()`, then requests JSON using
  `get_results(job_id=..., format="json")`. The SDK models Digitise results as
  `documents[].pages[]` with `file_name`, `page_number`, and `content`.
- Official docs list Marathi `mr-IN` and Hindi `hi-IN` as supported. The
  benchmark selects Marathi; this documents API language support, not measured
  accuracy on this dataset.
- Page-limit docs conflict: the Document Intelligence overview says 30 pages
  per Digitise ZIP/PDF, while the Sarvam Vision page and pricing page say 10.
  This runner conservatively caps every job at 10 pages (10 + 10 + 3).
- Listed Digitise pricing is ₹0.50/page, excluding GST. The estimate for 23
  pages is ₹11.50 plus GST if all pages are billable. Account credits,
  entitlements, and final billing are not verified; raw status snapshots expose
  API-reported usage counts when a live run occurs.
- Sarvam's privacy policy says API inputs/uploads/outputs may be used for model
  training unless the account opts out and lists user-configurable content
  retention (default 30 days after last access). The EULA says content is not
  used for training without explicit opt-in. These statements differ; the
  applicable account/workspace settings and contract were not inspected. Check
  them before uploading benchmark documents.

Official references: [Document Intelligence overview](https://docs.sarvam.ai/api/api-guides-tutorials/document-intelligence/overview),
[Sarvam Vision](https://docs.sarvam.ai/api/getting-started/models/sarvam-vision),
[Python SDK](https://docs.sarvam.ai/api/getting-started/sdks),
[pricing](https://docs.sarvam.ai/api/getting-started/pricing),
[privacy policy](https://www.sarvam.ai/privacy-policy),
[platform FAQ](https://docs.sarvam.ai/api/platform/faq), and
[EULA](https://www.sarvam.ai/eula).
