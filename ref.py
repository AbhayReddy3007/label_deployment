"""Serious Safety Profile: Main Orchestrator.

Re-exports constants and functions; exposes the single entry-point
run_serious_safety_assessment for external callers.
"""

from medical_potential.gcp_utils import (
    append_dimension_score_to_bigquery,
    upload_dimension_payload_cache_to_gcs,
    upload_dimension_report_pdf_to_gcs,
)
from medical_potential.config import (
    GEMINI_FLASH_PREVIEW_MODEL,
    SERIOUS_SAFETY_PROFILE_DIMENSION_NAME,
)
from .generate_report_and_rationale import (
    generate_prompt_safety_report_bytes,
    generate_serious_safety_rationale,
)
from .scoring_logic import score_serious_safety_profile
from .web_data_extraction import (
    determine_approval_market_status,
    extract_incremental_trials,
    extract_serious_safety_data,
    identify_regulatory_impact,
    identify_post_marketing_safety,
    summarize_sae_across_trials,
    classify_sae_expectedness_and_category,
    update_existing_sae_events_in_table,
    update_new_sae_events_to_summary_table,
    calculate_trial_weights_and_agg_sae,
)

from .bq_utils import delete_existing_trials, push_serious_safety_trials_to_bq, load_serious_safety_sae_summary


import logging

# Set root logger to WARNING to suppress most dependency logs
logging.basicConfig(level=logging.WARNING, format='[%(levelname)s] %(message)s')

# Set your package logger to INFO (or DEBUG)
logger = logging.getLogger("medical_potential.serious_safety_profile")
logger.propagate = False
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))
logger.handlers = [handler]

DIMENSION_NAME = SERIOUS_SAFETY_PROFILE_DIMENSION_NAME

# Gemini model configuration (centralized for the whole module)
APPROVAL_MARKET_STATUS_MODEL = GEMINI_FLASH_PREVIEW_MODEL
TRIAL_EXTRACTION_MODEL = GEMINI_FLASH_PREVIEW_MODEL
SAE_CLASSIFICATION_MODEL = GEMINI_FLASH_PREVIEW_MODEL
REGULATORY_IMPACT_MODEL = GEMINI_FLASH_PREVIEW_MODEL
POST_MARKETING_SAFETY_MODEL = GEMINI_FLASH_PREVIEW_MODEL
REPORT_WRITER_MODEL = GEMINI_FLASH_PREVIEW_MODEL


async def run_serious_safety_assessment(
    molecule_name: str,
    max_trials: int = None,
    generate_pdf_report: bool = True,
    incremental_mode: bool = True
) -> dict:
    """Run the full serious safety assessment pipeline.

    Executes a comprehensive 14-step (Steps 0-13) assessment for clinical trial safety:

    **Data Collection & Aggregation (Steps 0-5):**
        0. Determine whether the molecule is approved and marketed.
        1. Check for incremental clinical trial rows (optional incremental mode).
        2. Extract clinical trial SAE data for the molecule.
        3. Persist refreshed trial rows into the serious-safety trials table.
        4. Calculate trial weights and aggregate weighted SAE rates across trials.
        5. Identify unique serious adverse events and study occurrence counts.

    **Contextualization & Classification (Steps 6-8):**
        6. Classify each SAE by expectedness within drug class + safety severity category.
        7. Assess regulatory impact using post-approval regulatory communications.
        8. Review post-marketing serious safety signals (marketed drugs only).

    **Scoring (Dimension 5 Framework):**
        Applies multi-step deterministic scoring (1-5) based on trial banding, severity patterns,
        expectedness, post-marketing evidence, and critical safety overrides.

    **Report Generation (Optional):**
        Generates a PDF report summarising the score and rationale.

    Args:
        molecule_name: The drug / molecule name (e.g., "semaglutide").
        max_trials: Maximum number of trials to extract (None = no limit).
        generate_pdf_report: Whether to generate a PDF report and upload to GCS (default: True).
    """
    logger.info("[DIMENSION START] Running dimension '%s' for molecule '%s'", DIMENSION_NAME, molecule_name)
    
    # Step 0: Approval and marketed status
    logger.info("[SERIOUS_SAFETY] Step 0: Determine approval and marketed status")
    approval_market_status = await determine_approval_market_status(
        molecule_name=molecule_name,
        model_name=APPROVAL_MARKET_STATUS_MODEL,
    )
    is_approved = bool(approval_market_status.get("is_approved", False))
    is_marketed = bool(approval_market_status.get("is_marketed", False))

    # Step 1: Check for incremental clinical trial data
    logger.info("[SERIOUS_SAFETY] Step 1: Extract incremental/ full clinical trials data from the clinical efficacy table")
    incremental_trials = extract_incremental_trials(molecule_name,
                                                    incremental_mode)


    # Step 2: Extract trial data (skip if no incremental trials)
    if len(incremental_trials) == 0:
        logger.warning("[SERIOUS_SAFETY] No incremental trial rows for %s; skipping extraction and persistence", molecule_name)
        extraction = {"trials": [], "total_trials": 0, "completeness": 0}
        trials = []
    else:
        logger.info("[SERIOUS_SAFETY] Step 2: Extract clinical trial SAE data")
        extraction = await extract_serious_safety_data(
            molecule_name=molecule_name,
            max_trials=max_trials,
            extraction_model=TRIAL_EXTRACTION_MODEL,
            incremental_trials=incremental_trials,
        )
        trials = extraction.get("trials", [])

        # Step 3: Persist refreshed trial rows into the serious-safety trials table
        try:
            logger.info("[SERIOUS_SAFETY] Step 3: Persisting refreshed trial rows to BigQuery")
            if incremental_mode:
                delete_existing_trials(molecule_name, extraction)
            push_serious_safety_trials_to_bq(molecule_name, extraction, incremental_mode=incremental_mode)
            logger.info("[SERIOUS_SAFETY] Step 3: Persisted trial rows to BigQuery")
        except Exception as e:
            logger.exception("[SERIOUS_SAFETY] Failed to persist serious-safety trial rows for %s", molecule_name)
            return

    # Step 4-6: Summarize, update existing SAE table, and classify new SAEs (run only for incremental trials)
    if len(incremental_trials) > 0:
        # Step 4: Summarize all serious adverse events across all the trials
        logger.info("[SERIOUS_SAFETY] Step 4: Summarize SAE events across trials")
        sae_study_summary = summarize_sae_across_trials(trials)

        # Step 5: Update existing SAE events in summary table
        logger.info("[SERIOUS_SAFETY] Step 5: Update existing SAE events in summary table")
        try:
            existing_sae_items, new_sae_items = update_existing_sae_events_in_table(
                molecule_name=molecule_name,
                sae_summary=sae_study_summary,
            )
        except Exception:
            logger.exception("[SERIOUS_SAFETY] Failed to update existing SAE events in table for %s", molecule_name)
            existing_sae_items, new_sae_items = [], []

        # Step 6: Classify each NEW SAE event by expectedness and severity category
        logger.info("[SERIOUS_SAFETY] Step 6: Classify NEW SAE events expectedness and severity category")
        sae_event_categorization = await classify_sae_expectedness_and_category(
            sae_study_summary=new_sae_items,
            molecule_class="GLP-1",
            model_name=SAE_CLASSIFICATION_MODEL,
        )

        # Step 7: Insert newly classified SAE events into the SAE summary table
        logger.info("[SERIOUS_SAFETY] Step 7: Insert NEW SAE events into summary table")
        try:
            inserted_new_items = update_new_sae_events_to_summary_table(
                molecule_name=molecule_name,
                classified_sae_events=sae_event_categorization,
            )
        except Exception:
            logger.exception("[SERIOUS_SAFETY] Failed to insert new SAE events for %s", molecule_name)
            inserted_new_items = []
    else:
        logger.info("[SERIOUS_SAFETY] No incremental trials; skipping SAE summarization, update, and classification")
        # Load existing SAE event categorization from the SAE summary BigQuery table
        try:
            sae_event_categorization = load_serious_safety_sae_summary(molecule_name)
        except Exception:
            logger.exception("[SERIOUS_SAFETY] Failed to load SAE event categorization from BigQuery for %s", molecule_name)
            sae_event_categorization = []

    # Step 8: Aggregate weighted SAE rates (uses approval status for dosage weighting)
    logger.info("[SERIOUS_SAFETY] Step 8: Calculate trial weights and aggregate weighted SAE rates")
    trials_with_weights_added = calculate_trial_weights_and_agg_sae(molecule_name, is_approved=is_approved)

    # Step 9: Regulatory impact
    logger.info("[SERIOUS_SAFETY] Step 9: Assess regulatory impact")
    regulatory_impact = await identify_regulatory_impact(
        molecule_name=molecule_name,
        model_name=REGULATORY_IMPACT_MODEL,
    )

    # Step 10: Post-marketing safety signals (marketed drugs only)
    logger.info("[SERIOUS_SAFETY] Step 10: Evaluate post-marketing safety signals")
    if is_marketed:
        post_marketing_safety = await identify_post_marketing_safety(
            molecule_name=molecule_name,
            model_name=POST_MARKETING_SAFETY_MODEL,
        )
        post_marketing_safety["is_approved"] = is_approved
        post_marketing_safety["is_marketed"] = is_marketed
        post_marketing_safety["status_summary"] = approval_market_status.get("status_summary", "")
        post_marketing_safety["major_markets"] = approval_market_status.get("major_markets", [])
    else:
        post_marketing_safety = {
            "evaluated": False,
            "is_approved": is_approved,
            "is_marketed": is_marketed,
            "parse_error": False,
            "parse_error_stage": None,
            "skip_reason": "Post-marketing review runs only for marketed drugs",
            "status_summary": approval_market_status.get("status_summary", ""),
            "major_markets": approval_market_status.get("major_markets", []),
            "new_serious_risks": None,
            "new_serious_risks_rationale": None,
            "stronger_warning_added": None,
            "stronger_warning_added_rationale": None,
            "rems_added": None,
            "rems_added_rationale": None,
            "boxed_warning_added": None,
            "boxed_warning_added_rationale": None,
            "major_label_restriction": None,
            "major_label_restriction_rationale": None,
            "withdrawal_from_market": None,
            "withdrawal_from_market_rationale": None,
            "risk_level": "unknown",
            "key_findings": [],
            "grounding_summary": "",
            "sources_summary": [],
            "raw_post_marketing_response": "",
        }


    # Step 11: Score the drug using the Dimension 5 framework
    logger.info("[SERIOUS_SAFETY] Step 11: Compute final serious safety score")
    safety_score = score_serious_safety_profile(
        sae_aggregation=trials_with_weights_added,
        regulatory_impact=regulatory_impact,
        post_marketing_safety=post_marketing_safety,
        is_marketed=is_marketed,
        molecule_name=molecule_name,
    )
    logger.info("[SERIOUS_SAFETY] Completed assessment with score=%s/5", safety_score["score"])

    # preparing the input payload for the report

    # Getting the info on the total number of trials used
    total_trials = None
    try:
        if isinstance(trials_with_weights_added, dict):
            t = trials_with_weights_added.get("trials")
            if isinstance(t, (list, tuple)):
                total_trials = len(t)
    except Exception:
        total_trials = None

    report_input_payload = {
        "molecule_name": molecule_name,
        "approval_market_status": approval_market_status,
        "sae_aggregation": trials_with_weights_added,
        "sae_event_categorization": sae_event_categorization,
        "regulatory_impact": regulatory_impact,
        "post_marketing_safety": post_marketing_safety,
        "safety_score": safety_score,
    }

    # Step 12: Optional - Generate PDF report
    report_gcs_uri = None
    report_content = {
        "summary_table": {
            "score": safety_score.get("score"),
            "score_text": f"{safety_score.get('score')}/5" if safety_score.get("score") is not None else "N/A",
            "total_trials": total_trials,
        },
        "sections": {},
    }
    report_prompt_payload = None
    if generate_pdf_report:
        logger.info("[SERIOUS_SAFETY] Step 12: Generating PDF report")

        try:
            # Generate PDF in memory and capture structured section content
            report_result = await generate_prompt_safety_report_bytes(
                report_input_payload,
                model_name=REPORT_WRITER_MODEL,
            )
            if isinstance(report_result, tuple):
                # Support either (pdf_bytes, report_content) or (pdf_bytes, report_content, payload)
                if len(report_result) == 2:
                    pdf_bytes, report_content = report_result
                    report_prompt_payload = None
                else:
                    pdf_bytes, report_content, report_prompt_payload = report_result
            else:
                pdf_bytes = report_result

            # Upload to GCS via shared utility
            report_gcs_uri, _ = upload_dimension_report_pdf_to_gcs(
                pdf_bytes=pdf_bytes,
                molecule_name=molecule_name,
                dimension_name=DIMENSION_NAME,
            )
        except Exception as e:
            logger.warning("[SERIOUS_SAFETY] Failed to generate PDF report: %s", e)

    # Step 13: Generate the rationale for the score
    logger.info("[SERIOUS_SAFETY] Step 13: Generating concise score rationale")
    rationale_prompt_payload = None
    rationale, rationale_prompt_payload = await generate_serious_safety_rationale(
        data=report_input_payload,
        model_name=REPORT_WRITER_MODEL,
    )

    # Step 14: Upload the rationale and score to BQ
    logger.info("[SERIOUS_SAFETY] Step 14: Uploading score and rationale to dim_scores")
    append_dimension_score_to_bigquery(
        molecule_name=molecule_name,
        dimension_name=DIMENSION_NAME,
        score=safety_score.get("score"),
        rationale=rationale,
    )

    # Step 15: Uploading the output payload to GCS pipeline cache
    output_payload = {
        "molecule_name": molecule_name,
        "approval_market_status": approval_market_status,
        "total_trials": total_trials,
        "sae_aggregation": trials_with_weights_added,
        "sae_event_categorization": sae_event_categorization,
        "regulatory_impact": regulatory_impact,
        "post_marketing_safety": post_marketing_safety,
        "safety_score": safety_score,
        "report": report_content,
        "rationale": rationale,
        "report_gcs_uri": report_gcs_uri,
        "report_and_rationale_input_payload": report_input_payload,
        "report_prompt_payload": report_prompt_payload,
        "rationale_prompt_payload": rationale_prompt_payload,
    }

    logger.info("[SERIOUS_SAFETY] Step 15: Cache final payload")
    output_payload = dict(output_payload)
    cache_gcs_uri = upload_dimension_payload_cache_to_gcs(
        payload=output_payload,
        molecule_name=molecule_name,
        dimension_name=DIMENSION_NAME,
    )

    output_payload["cache_gcs_uri"] = cache_gcs_uri

    logger.info("[DIMENSION COMPLETE] Dimension '%s' complete for molecule '%s'", DIMENSION_NAME, molecule_name)

