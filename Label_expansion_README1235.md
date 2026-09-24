# Label Expansion Opportunity Pipeline

The `label_expansion()` pipeline identifies and scores potential label-expansion opportunities for a drug — new indications beyond its current approved use where clinical evidence, regulatory signals, and biological rationale suggest the drug could be effective. It ingests data from clinical trials, FDA labels, and public pipeline intelligence, maps each opportunity to a validated disease in Open Targets, scores it against a composite evidence model, and produces a business-facing PDF report with a ranked opportunity landscape.

---

## Pipeline Overview

```text
Step 1   Discovery
         ├── 1.1  Clinical Trial Analysis      → raw trial-indication records (BigQuery)
         ├── 1.2  FDA Label Analysis            → approved indication records (BigQuery)
         ├── 1.3  Web / Public Signal Analysis  → pipeline signal records (BigQuery)
         └── 1.4  Indication Filtering          → cleaned indication list (BigQuery)
                    │
                    ▼
Step 2   Filter, Merge & Push
         De-duplicate and upsert all indications → label_expansion table (BigQuery)
                    │
                    ▼
Step 3   MOA → Open Targets Target Mapping
         Resolve drug target → Ensembl ID (BigQuery)
                    │
                    ▼
Step 4   Indication → Open Targets Disease Mapping
         Map each secondary indication to a validated Open Targets disease ID (BigQuery)
                    │
                    ▼
Step 5   Data Fetching & Trial Selection
         Enrich each indication with trial metadata → best-trial record per indication (BigQuery)
                    │
                    ▼
Step 6   Score Calculation
         Compute composite Final Score → scored_indications table (BigQuery)
                    │
                    ▼
Step 7   Rationale & PDF Report Generation
         → PDF report (GCS) + JSON payload (GCS) + dimension score (BigQuery)
```

> **Primary indications** (i.e. the drug's established, approved uses) are discovered and recorded but are not scored. Only **Secondary indications** — emerging or unapproved uses with active clinical or commercial interest — proceed through scoring.

---

## Step 1 — Discovery

The pipeline casts a wide net across three independent data sources to discover every credible indication the drug may be associated with. All discovery outputs are written to BigQuery as raw indication records before filtering.

### 1.1 Clinical Trial Analysis

Identifies indications being actively studied in registered clinical trials.

- Pulls registered trial records for the drug from the BigQuery trials table.
- Pre-filters trials using an LLM confidence score to discard low-signal registrations before any API call is made.
- Sends each remaining trial through a Gemini + Google Search analysis that reads the trial's title, conditions, and interventions to extract the disease or indication under investigation.
- Classifies each extracted indication as `Primary` (approved use) or `Secondary` (expansion candidate) and generates an Open Targets-compatible disease name hint used in Step 4.
- Supports incremental runs: trials already processed in a prior run are detected and skipped, so the pipeline can resume or top-up without re-processing the full trial corpus.

**Artifacts produced:** raw trial-indication rows written to the `label_expansion_trials` BigQuery table.

### 1.2 FDA Label Analysis

Captures the drug's current approved indication footprint from the regulator's own language.

- Queries the openFDA REST API using the drug's known brand names to retrieve the current approved label.
- Extracts the `indications_and_usage` section verbatim from the FDA label.
- Passes this section to Gemini, which parses and normalizes the approved indications and classifies each as `Primary` or `Secondary`.

**Artifacts produced:** FDA-sourced indication rows written to the `label_expansion_fda` BigQuery table.

### 1.3 Web / Public Signal Analysis

Surfaces forward-looking expansion signals not yet visible in the trial registry or the approved label.

- Uses Gemini with Google Search grounding to scan publicly available commercial and scientific intelligence, including investor presentations, press releases, earnings calls, and pipeline databases.
- Identifies indications where the drug is being positioned or discussed as a potential expansion candidate, even before a trial has been registered.

**Artifacts produced:** web-signal indication rows written to the `label_expansion_web` BigQuery table.

### 1.4 Indication Filtering

Cleans and consolidates the three discovery streams into a single, validated list of genuine disease indications.

- Merges outputs from Steps 1.1–1.3 into a unified candidate list.
- Applies an LLM-based filter to remove extractions that are not disease indications: trial endpoints (e.g. HbA1c reduction), biomarkers (e.g. PD-L1 expression), pharmacokinetic parameters (e.g. bioavailability), and medical procedures.
- The filtered list is what enters Step 2 for de-duplication and persistence.

**Artifacts produced:** cleaned indication list, ready for upsert.

---

## Step 2 — Filter, Merge & Push

Consolidates all discovery outputs into a single authoritative indication table.

- Merges the filtered indications from clinical trials, FDA labels, and web signals.
- De-duplicates records on the composite key of drug name + indication + trial ID, so the same indication appearing in multiple sources is represented once.
- Resolves source conflicts by preferring trial-sourced records over web-sourced records, since trial data carries more regulatory weight.
- Upserts the resulting de-duplicated indication set into the master `label_expansion` BigQuery table.

**Artifacts produced:** upserted rows in the `label_expansion` BigQuery table — the canonical indication record used by all downstream steps.

---

## Step 3 — MOA → Open Targets Target Mapping

Establishes the biological anchor needed to query Open Targets for evidence.

- Retrieves the drug's Mechanism of Action (MOA) from the drug details table in BigQuery.
- Uses the MOA to resolve the drug's primary molecular target within the Open Targets knowledge graph.
- Retrieves the corresponding Ensembl gene ID for that target — the identifier used by Open Targets to link targets to diseases.
- Stores the MOA-to-Ensembl mapping in BigQuery for use in Steps 4 and 5.

**Artifacts produced:** MOA-to-Ensembl ID mapping row in BigQuery, used as the biological reference point for all disease association queries.

---

## Step 4 — Indication → Open Targets Disease Mapping

Translates each secondary indication — often expressed in free-text clinical language — into a validated, machine-readable Open Targets disease identifier. This is the most complex step in the pipeline and runs through a layered resolution strategy.

### 4.1 Normalization

- Standardizes the raw indication text before any matching is attempted: corrects spelling variants, handles hyphenation and parenthetical qualifiers, and collapses common synonyms to a canonical form.

### 4.2 Semantic Clustering

- Groups indications that describe the same disease under a single canonical name (e.g. "T2DM", "Type 2 Diabetes", and "adult-onset diabetes" are treated as one disease).
- Reduces duplicate Open Targets queries and improves mapping consistency across sources.

### 4.3 Direct Match (Open Targets Search)

- Before any Gemini reasoning is invoked, performs a direct Open Targets text search using the raw indication text (or its LLM-generated disease-name hint from Step 1.1).
- If the disease name returned by Open Targets is an exact, case-insensitive match to the indication text, the mapping is accepted immediately — no LLM call is needed for this case.
- This resolves clear-cut, unambiguous indications at minimal cost and latency, leaving only genuinely ambiguous cases for Path A.

### 4.4 Path A — Target-Disease Matching

- For indications not resolved by the direct match, uses Gemini to match each normalized indication against the Open Targets target-disease association table for the drug's Ensembl target ID.
- This is the highest-confidence remaining path: if the indication already has an established association with the drug's target in Open Targets, the match is direct and biologically grounded.

### 4.5 Path B — Open Targets Text Search

- For indications not resolved by Path A, performs a text search against the Open Targets disease ontology using the LLM-generated disease name hint from Step 1.1 as the search query.
- Ranks candidates and selects the closest match by name and ontology context. This is a deterministic scoring step — no Gemini call is made here.

### 4.6 Path C — Gemini + Google Search

- The fallback path for indications that neither the direct match, Path A, nor Path B can resolve with confidence.
- Uses Gemini with Google Search grounding to identify the correct Open Targets disease entry — effectively asking the model to reason about which ontology term best captures the clinical meaning of the indication.

### 4.7 Validation

- All candidate mappings from Path A, Path B, and Path C are validated before being accepted (direct matches, being exact by construction, skip this check).
- A Gemini-based validation step rejects false positives caused by surface-level word overlap (e.g. mapping "diabetic nephropathy" to "nephropathy" simply because both contain the word).
- Indications that cannot be mapped with sufficient confidence remain `null` in the table and are automatically retried in the next pipeline run.

**Artifacts produced:** Open Targets disease ID, disease name, and association score written to the `label_expansion` BigQuery table for each secondary indication.

---

## Step 5 — Data Fetching & Trial Selection

Enriches each mapped secondary indication with the trial-level evidence needed for scoring.

### 5.1 Data Fetching

For each secondary indication with a confirmed Open Targets disease mapping:

- Retrieves the full set of associated trial records from BigQuery.
- Enriches each trial record with fields required for scoring: geographic region, patient sample size, and dosage information.
- Where these fields are missing from the registry, uses Gemini + Google Search to retrieve them from public sources (e.g. trial results pages, published protocols).
- Queries the Open Targets API for the association score between the drug's Ensembl target and the mapped disease — this score reflects the biological plausibility of the drug-disease link and is a key input to the scoring model.
- Caches all fetched data in BigQuery to avoid redundant API and search calls on subsequent runs.

**Artifacts produced:** enriched trial records and Open Targets association scores in the `label_expansion_trials` BigQuery table.

### 5.2 Trial Weight Calculation

Each trial is assigned a composite weight that reflects its evidentiary strength:

- **Clinical phase** — Phase 3 and approved trials carry the most weight; Phase 1 the least.
- **Geography** — Trials conducted in regulated Tier-1 markets (US, EU, UK) are weighted more heavily than trials in other regions.
- **Sample size** — Larger trials provide stronger evidence and receive a proportionally higher weight.
- **Dosage confidence** — Trials with a confirmed, on-label dosage receive a higher weight than those with uncertain or off-label dosing.

Trials with an `Approved` status bypass the weighting model and are assigned a fixed trial weight of `1.0` — the maximum possible value — reflecting that regulatory approval is the strongest available evidence signal.

### 5.3 Best Trial Selection

- Constructs a composite key (`ta_i`) from the therapy area and the Open Targets disease ID to ensure each disease-within-therapy-area is scored independently.
- Removes any rows where no valid Open Targets disease ID was resolved (Step 4 null cases).
- For each `ta_i` key, selects the single highest-weighted trial as the representative evidence record to avoid double-counting multiple trials for the same indication.

**Artifacts produced:** one best-trial record per indication, ready for the scoring model.

---

## Step 6 — Score Calculation

Computes a composite **Final Score** (range: 1–5) for each label-expansion opportunity, combining biological plausibility, clinical evidence strength, and the breadth and coherence of the overall expansion opportunity across indications and therapy areas.

The calculation proceeds through the following stages:

| # | Component | What it captures |
|---|-----------|-----------------|
| 1 | `prior` | Open Targets association score — the baseline biological plausibility of the drug-disease link |
| 2 | `maturity_weight` | How far along the clinical development path the leading trial has progressed |
| 3 | `effective_indications` | The count of indications with sufficient evidence to contribute meaningfully to the score |
| 4 | `effective_therapy_areas` | The number of distinct therapy areas represented across effective indications |
| 5 | `Q_i` | Combined trial-quality weight for indication *i*, aggregating phase, geography, size, and dosage signals |
| 6 | `e_phase_i` | Phase-specific evidence scaling factor |
| 7 | `e_i = Q_i × e_phase_i` | Overall evidence strength for indication *i* |
| 8 | `link = 1 − (1−prior)(1−e_i)` | Bayesian-style combination of biological prior and clinical evidence |
| 9 | `link_ta` | Maturity-weighted average of `link` values across all indications within a therapy area |
| 10 | `B_ind` | Indication breadth — how many indications contribute meaningfully |
| 11 | `B_ta` | Therapy-area breadth — how many therapy areas contribute meaningfully |
| 12 | `B` | Overall breadth — combined indication and therapy-area spread |
| 13 | `coherence` | How closely related the indications are across therapy areas (a focused cluster scores higher than scattered disease areas) |
| 14 | Coherence adjustment | `C = 0.1 + 0.9 × coherence^1.75` — softens the penalty for moderate coherence while rewarding tightly focused expansion profiles |
| 15 | **Final Score** | `Final Score = 1 + 4 × B × C` — scales to 1–5, where 5 represents the broadest, most coherent, and best-evidenced expansion opportunity |

All intermediate values and the full formula breakdown are stored alongside the Final Score in the `scored_indications` BigQuery table for auditability and re-scoring.

**Artifacts produced:** Final Score, all intermediate components, and formula breakdown in the `scored_indications` BigQuery table.

---

## Step 7 — Rationale & PDF Report Generation

Translates the scored opportunity landscape into a business-ready narrative and formal report.

### 7.1 Rationale Generation

- Uses Gemini to generate a concise, plain-English rationale for each scored opportunity — a short summary of why the expansion is scientifically and commercially credible.
- Rationale is capped at 50 words and written for a non-technical business audience: it references clinical signals and therapy-area positioning rather than internal field names or scoring formula components.

### 7.2 PDF Report Generation

The PDF report is structured in four sections:

**1. Summary snapshot**
A one-page overview of the expansion landscape: number of therapy areas identified, total secondary indications scored, and the top Final Score achieved.

**2. Business Narrative** *(Gemini-generated)*
A structured analytical narrative covering:
- **Headline** — the single most important commercial takeaway
- **Indication Landscape** — quantitative context on the size and shape of the opportunity set
- **Key Insights** — 4–6 specific findings with their business implications (evidence strength, competitive positioning, regulatory pathway readiness, geographic coverage)
- **Evidence Gaps & Risks** — material gaps in clinical data that represent commercial or regulatory risk
- **Bottom Line** — a direct, actionable recommendation for decision-makers

**3. Expansion Indications Table**
A structured table of all scored secondary indications, grouped by therapy area, showing indication name, Final Score, Open Targets disease mapping, and leading trial details.

**4. Methodology**
A transparent breakdown of the scoring model for technical and regulatory reviewers, including the breadth calculation, coherence calculation, Final Score formula, and the full scored-indication table with all intermediate values.

### 7.3 Persistence

The following artifacts are stored at the end of every pipeline run:

| Artifact | Location | Contents |
|----------|----------|----------|
| PDF report | GCS | Full business-facing report |
| Archived PDF | GCS | Timestamped copy for version history |
| JSON payload | GCS | Complete pipeline output including all scores, rationales, and metadata |
| Archived JSON | GCS | Timestamped copy |
| Dimension score | BigQuery | Final Score and rationale for dashboard consumption |

---

## Step Input / Output Reference

| Step | Input | Output |
|------|-------|--------|
| 1.1 Clinical Trial Analysis | Drug name; registered trial records (BigQuery trials table), pre-filtered by LLM confidence score | Trial-sourced indication rows: indication name, Primary/Secondary classification, Open Targets disease-name hint |
| 1.2 FDA Label Analysis | Drug name; openFDA brand lookup → raw `indications_and_usage` label text | FDA-sourced indication rows: indication name, Primary/Secondary classification |
| 1.3 Web / Public Signal Analysis | Drug name | Web-sourced indication rows: indication name, Primary/Secondary classification, supporting rationale |
| 1.4 Indication Filtering | Combined raw indication rows from Steps 1.1–1.3 | Cleaned indication list, with trial endpoints / biomarkers / PK parameters / procedures removed |
| 2. Filter, Merge & Push | Cleaned indication list from Step 1.4 | De-duplicated rows upserted into the `label_expansion` BigQuery table |
| 3. MOA → Target Mapping | Drug's Mechanism of Action (drug details BigQuery table) | MOA → Open Targets target name + Ensembl gene ID |
| 4.1 Normalization | Raw indication text from `label_expansion` table | Standardized indication text (spelling, hyphenation, parentheticals, common synonyms collapsed) |
| 4.2 Semantic Clustering | Normalized indications for the drug | Indications grouped under one canonical name per distinct disease |
| 4.3 Direct Match (OT Search) | Canonical indication name (or its OT name hint) | Confirmed Open Targets disease ID for exact-name matches; unresolved indications pass through |
| 4.4 Path A — Target-Disease Matching | Indications unresolved by Direct Match; Ensembl target ID from Step 3 | Open Targets disease ID for indications matched to the drug's known target-disease associations |
| 4.5 Path B — Open Targets Text Search | Indications unresolved by Path A; OT disease-name hint | Best-scoring Open Targets disease ID from ranked text-search candidates |
| 4.6 Path C — Gemini + Google Search | Indications unresolved by Path A and Path B | Open Targets disease ID identified via grounded web reasoning |
| 4.7 Validation | All candidate mappings from Paths A–C | Confirmed mappings (false-positive word-overlap matches rejected and left `null` for retry) |
| 5.1 Data Fetching | Secondary indications with a confirmed Open Targets disease mapping | Enriched trial records (region, sample size, dosage) + Open Targets association score, cached in BigQuery |
| 5.2 Trial Weight Calculation | Enriched trial records from Step 5.1 | Composite `trial_weight` per trial (phase, geography, sample size, dosage confidence) |
| 5.3 Best Trial Selection | Weighted trial records, keyed by `ta_i` (therapy area + OT disease ID) | One best-trial record per indication |
| 6. Score Calculation | Best-trial records from Step 5.3 | Final Score (1–5) and full formula breakdown per indication, written to `scored_indications` |
| 7.1 Rationale Generation | Scored indications for the drug | Plain-English rationale (≤50 words) per opportunity set |
| 7.2 PDF Report Generation | Scored indications, Step 7.1 rationale | Structured PDF report (summary snapshot, business narrative, expansion indications table, methodology) |
| 7.3 Persistence | Generated PDF + JSON payload | PDF + archived PDF (GCS), JSON payload + archived JSON (GCS), dimension score (BigQuery) |

---

## Gemini Usage

Gemini is used at nine distinct points in the pipeline. Each call is purpose-built for a specific reasoning or retrieval task that structured data queries alone cannot perform. Note that Step 4's Direct Match (4.3) and Path B (4.5) are deliberately **not** Gemini calls — they're cheaper, deterministic text-search steps that only escalate to Gemini when they can't resolve a mapping with confidence.

| Step | Task | Gemini capability used |
|------|------|----------------------|
| 1.1 Clinical Trial Analysis | Extract the disease indication from a trial's title, conditions, and interventions; classify as Primary or Secondary; generate an Open Targets disease name hint | Text extraction + classification, optionally grounded with Google Search |
| 1.2 FDA Label Analysis | Parse the raw `indications_and_usage` label text and extract normalized, classified indications | Text extraction + classification |
| 1.3 Web / Public Signal Analysis | Scan investor materials, press releases, and pipeline databases for forward-looking expansion signals not yet in the trial registry | Grounded generation with Google Search |
| 4.2 Semantic Clustering | Group indications describing the same disease under one canonical name (e.g. "T2DM" / "Type 2 Diabetes" / "adult-onset diabetes") before any Open Targets matching is attempted | Text clustering / reasoning |
| 4.4 Path A — Target-Disease Matching | Match a normalized indication against the Open Targets target-disease association list for the drug's Ensembl target ID | Structured matching / reasoning |
| 4.6 Path C — Disease Mapping | Resolve an indication to its correct Open Targets disease entry when the direct match, Path A, and Path B all fail | Grounded reasoning with Google Search |
| 4.7 Validation | Reject candidate disease mappings (from Path A/B/C) that are only superficial word-overlap matches rather than genuine clinical matches | Structured verification / reasoning |
| 5.1 Data Fetching | Retrieve missing trial metadata (region, sample size, dosage) from public sources when the trial registry record is incomplete | Grounded retrieval with Google Search |
| 7.1 / 7.2 Report Generation | Generate the plain-English rationale (50-word cap) and the full structured business narrative (Headline, Indication Landscape, Key Insights, Evidence Gaps, Bottom Line) for the PDF report | Long-form structured generation |

---

## Pipeline Re-entry

The pipeline supports re-entry from any stage, allowing specific steps to be rerun without repeating earlier work. This is useful when, for example, the scoring model changes but the indication mapping does not need to be redone, or when new trials are added to the registry and only the discovery step needs to be refreshed.

```text
discovery
    ↓
moa_mapping
    ↓
indication_mapping
    ↓
scoring
    ↓
report_generation
```

Each re-entry point reads from the BigQuery artifacts written by the preceding step, so intermediate results are fully preserved and reusable across runs.
