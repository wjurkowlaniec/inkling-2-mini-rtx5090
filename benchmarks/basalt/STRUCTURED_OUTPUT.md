# Structured JSON: what is actually faster?

Measured **2026-11-25** on one RTX 5090, Basalt 0.5.7, the uncensored checkpoint converted to Basalt, DSpark cap 8, FP8 KV, two 131,072-token slots, Vision enabled. All requests used `reasoning_effort: none`, natural EOS and `max_tokens: 3072`; the server was already warm. Output tokens include any reasoning. These short synthetic cases are not a general-quality or real-agent benchmark.

## Output: `json_object` versus JSON requested in plain text

We sent **the identical prompt and data** with and without `response_format: {"type":"json_object"}`. The first task extracted 24 inventory records from a text note; the second generated 24 fictional records. Every result was parsed as JSON and checked for 24 correctly ordered IDs, fields, types, categories and value ranges; extraction additionally checked every value against the input. This is a specific content check, not a test of arbitrary schemas or factual accuracy. Three batches per condition, mode order alternated, temperature 0 and 0.7; the same short prompts may hit prefix cache.

| Task, temperature | Mode | 1 request: median time | 1 request: median engine decode | Median output tokens | 2 requests: median batch time | Valid single / paired requests |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Extraction, 0 | Prompt-only | 1.665 s | 740 tok/s | 1,220 | 2.940 s | 3/3, 6/6 |
| Extraction, 0 | `json_object` | 0.790 s | 781 tok/s | 611 | 1.678 s | 3/3, 6/6 |
| Generation, 0 | Prompt-only | 1.392 s | 670 tok/s | 928 | 2.685 s | 3/3, 6/6 |
| Generation, 0 | `json_object` | 0.900 s | 684 tok/s | 611 | 1.661 s | 3/3, 6/6 |
| Extraction, 0.7 | Prompt-only | 1.448 s | 703 tok/s | 1,046 | 2.869 s | 3/3, 6/6 |
| Extraction, 0.7 | `json_object` | 0.798 s | 772 tok/s | 611 | 1.714 s | 3/3, 6/6 |
| Generation, 0.7 | Prompt-only | 1.407 s | 652 tok/s | 913 | 2.681 s | 3/3, 6/6 |
| Generation, 0.7 | `json_object` | 0.901 s | 683 tok/s | 611 | 1.717 s | 3/3, 6/6 |

For extraction at temperature 0, `json_object` reduced median wall time by about **53%** (2.1× faster completion), but its median decode rate rose by only **5.6%**. In these runs the Basalt JSON-mode responses had **zero reasoning characters**, while prompt-only responses contained reasoning; output token count dropped from 1,220 to 611. Thus the completion-time difference is mostly fewer generated tokens, not a 2× faster GPU. `reasoning_effort: none` is not a guarantee that prompt-only requests emit no reasoning. JSON mode also changes the decoding constraint and can affect output; use it when the task needs an object, not as a universal speed flag. Neither this test nor JSON syntax guarantees semantic accuracy for arbitrary tasks.

Single-request decode uses Basalt's `timings.predicted_per_second`, a request-level decode rate, not a standalone GPU-kernel measurement. Two-request figures use wall time for both results and include input processing, waiting and scheduling. Seeded temperature 0.7 can produce different output; one or two requests may receive different execution time. Do not compare a two-request per-stream decode number to aggregate throughput.

## JSON object versus strict JSON Schema

A separate run compared `json_object` with `json_schema` and `strict: true`, on the same two prompts, three alternating batches per condition, temperatures 0 and 0.7. The schema constrained the object, array, fields, types and category enum; it did **not** enforce 24 records or correct extraction values. Those were checked separately.

| Task, temperature | JSON mode | Strict schema | Two requests: JSON mode / strict schema |
| --- | ---: | ---: | ---: |
| Extraction, 0 | 0.784 s / 789 tok/s | 0.782 s / 788 tok/s | 1.669 / 1.680 s |
| Generation, 0 | 0.899 s / 685 tok/s | 0.893 s / 690 tok/s | 1.674 / 1.528 s |
| Extraction, 0.7 | 0.795 s / 775 tok/s | 0.795 s / 775 tok/s | 1.703 / 1.702 s |
| Generation, 0.7 | 0.917 s / 671 tok/s | 0.914 s / 674 tok/s | 1.694 / 1.708 s |

Single-request cells show median request wall time and engine decode rate; pair cells show median batch wall time. All **72 requests** finished naturally and passed the task-specific data checks. The modes were essentially tied for one request in this small sample; the paired differences should not be treated as a stable advantage without more runs. Schema enforcement improves structural guarantees but does not establish factual correctness. Full runs: [one request](structured-schema-c1-2026-11-25.json), [two requests](structured-schema-c2-2026-11-25.json).

## Input: equivalent note versus JSON data

We supplied the same 24 inventory records in a compact JSON array or as text lines, requested the same output, and enabled `json_object` in both cases. Three batches per condition; the two input formats ran sequentially rather than interleaved, so small differences can be noise.

| Input, temperature | Median prompt tokens | 1 request: median wall | 1 request: median engine decode | 2 requests: median batch wall | Expected contents |
| --- | ---: | ---: | ---: | ---: | ---: |
| Text, 0 | 308 | 0.781 s | 794 tok/s | 1.548 s | 3/3, 6/6 |
| JSON, 0 | 670 | 0.789 s | 781 tok/s | 1.699 s | 3/3, 6/6 |
| Text, 0.7 | 308 | 0.793 s | 778 tok/s | 1.695 s | 3/3, 6/6 |
| JSON, 0.7 | 670 | 0.797 s | 773 tok/s | 1.733 s | 3/3, 6/6 |

On this small prompt, **JSON input showed no speed advantage**; it used more prompt tokens. This does not imply text is always better: format and length affect prefill, and model behavior can change. The output in all these checks was 611 tokens per request and contained the correct records.

## Reproduce

The tested server is a running [Basalt setup](../../basalt/README.md). `json_object` constrains JSON syntax, not the semantic schema; validate the result in your application. These commands create inference load, use synthetic input and write results locally:

```bash
python benchmarks/basalt/structured_output_benchmark.py --iterations 3 --concurrency 1 --output structured-c1.local.json
python benchmarks/basalt/structured_output_benchmark.py --iterations 3 --concurrency 2 --output structured-c2.local.json
python benchmarks/basalt/structured_output_benchmark.py --input-comparison --iterations 3 --concurrency 1 --output structured-input-c1.local.json
python benchmarks/basalt/structured_output_benchmark.py --input-comparison --iterations 3 --concurrency 2 --output structured-input-c2.local.json
python benchmarks/basalt/structured_output_benchmark.py --schema-comparison --iterations 3 --concurrency 1 --output structured-schema-c1.local.json
python benchmarks/basalt/structured_output_benchmark.py --schema-comparison --iterations 3 --concurrency 2 --output structured-schema-c2.local.json
```

Raw prompts, outputs, timing fields, validity checks and individual runs: [output, one](structured-c1-2026-11-25.json), [output, two](structured-c2-2026-11-25.json), [input, one](structured-input-c1-2026-11-25.json), [input, two](structured-input-c2-2026-11-25.json). All 108 requests completed and passed these specific checks.

Example request:

```json
{
  "model": "inkling",
  "messages": [{"role": "user", "content": "Return a JSON object with a city and temperature in Celsius."}],
  "response_format": {"type": "json_object"},
  "reasoning_effort": "none",
  "max_tokens": 1024
}
```

The earlier **706 tok/s** number in the README is a separate `basalt-bench` standard, single-sequence, 8,192-token highly predictable JSON fixture. It does not measure this API JSON-mode comparison. See [standard benchmark](README.md#basalt-standard-engine-benchmark).
