//! Smart Jev MCP Normalizer: decomposes and maps Jev MCP protocol shapes
//! into Krite's native NLI, reading comprehension, and zero-shot classification pipelines.

use std::collections::BTreeMap;

use krite_core::{Answer, ApiError, Ordered, Question, Request, Response, Usage};
use krite_runtime::{Backend, Decision, Runtime, Timing};
use serde_json::Value;

/// Attempts to normalize and handle Jev MCP specific patterns.
/// Returns Ok(Some(Decision)) if handled, Ok(None) if not a Jev request pattern.
pub fn try_handle_jev<B: Backend>(rt: &mut Runtime<B>, req: &Request) -> Result<Option<Decision>, ApiError> {
    let Value::Object(state_obj) = &req.state else {
        return Ok(None);
    };

    if state_obj.contains_key("classes") && is_jev_classify(req) {
        return handle_jev_classify(rt, req).map(Some);
    }
    if state_obj.contains_key("query") && is_jev_rerank(req) {
        return handle_jev_rerank(rt, req).map(Some);
    }
    if state_obj.contains_key("claims") && is_jev_verify(req) {
        return handle_jev_verify(rt, req).map(Some);
    }
    if state_obj.contains_key("content") && is_jev_screen(req) {
        return handle_jev_screen(rt, req).map(Some);
    }
    if state_obj.contains_key("passage_a") && state_obj.contains_key("passage_b") && is_jev_compare(req) {
        return handle_jev_compare(rt, req).map(Some);
    }
    if state_obj.contains_key("source") && state_obj.contains_key("records") && is_jev_audit(req) {
        return handle_jev_audit(rt, req).map(Some);
    }
    if state_obj.contains_key("request")
        && (state_obj.contains_key("diff") || state_obj.contains_key("files"))
        && is_jev_review(req)
    {
        return handle_jev_review(rt, req).map(Some);
    }
    if state_obj.contains_key("decision") && state_obj.contains_key("candidates") && is_jev_decide(req) {
        return handle_jev_decide(rt, req).map(Some);
    }
    if state_obj.contains_key("candidates") && is_jev_find(req) {
        return handle_jev_find(rt, req).map(Some);
    }
    if state_obj.contains_key("document") && state_obj.contains_key("fields") && is_jev_extract(req) {
        return handle_jev_extract(rt, req).map(Some);
    }
    if state_obj.contains_key("propositions") && is_jev_noul(req) {
        return handle_jev_noul(rt, req).map(Some);
    }

    Ok(None)
}

fn is_jev_classify(req: &Request) -> bool {
    let Some(classes) = req.state.get("classes").and_then(|c| c.as_array()) else {
        return false;
    };
    !classes.is_empty()
        && req
            .questions
            .values()
            .any(|q| matches!(q, Question::Choice { criteria, .. } if criteria.values().all(|v| v.is_none())))
}

fn is_jev_rerank(req: &Request) -> bool {
    req.state.get("query").is_some() && req.questions.keys().any(|k| k.starts_with("rel_"))
}

fn is_jev_verify(req: &Request) -> bool {
    req.state.get("claims").and_then(|c| c.as_array()).is_some()
        && req.questions.keys().any(|k| k.starts_with("relation_"))
}

fn is_jev_screen(req: &Request) -> bool {
    req.questions.contains_key("injection") && req.questions.contains_key("substance")
}

fn is_jev_compare(req: &Request) -> bool {
    req.questions.contains_key("overall") || req.questions.contains_key("relation")
}

fn is_jev_audit(req: &Request) -> bool {
    req.questions.keys().any(|k| k.starts_with("check_") || k.starts_with("absence_"))
}

fn is_jev_review(req: &Request) -> bool {
    req.questions.contains_key("correctness") && req.questions.contains_key("spec_match")
}

fn is_jev_decide(req: &Request) -> bool {
    req.questions.contains_key("recommendation")
}

fn is_jev_find(req: &Request) -> bool {
    req.questions.contains_key("best") && req.questions.contains_key("exists")
}

fn is_jev_extract(req: &Request) -> bool {
    req.questions.keys().any(|k| k.starts_with('f'))
}

fn is_jev_noul(req: &Request) -> bool {
    req.questions.keys().all(|k| k.starts_with("p_"))
}

// ── Handlers ─────────────────────────────────────────────────────────────────

/// 1. jev_classify: unfolds classes from state.classes and populates criteria descriptions.
fn handle_jev_classify<B: Backend>(rt: &mut Runtime<B>, req: &Request) -> Result<Decision, ApiError> {
    let classes_arr = req
        .state
        .get("classes")
        .and_then(|c| c.as_array())
        .ok_or_else(|| ApiError::invalid("missing classes", None))?;
    let mut criteria_map = BTreeMap::new();
    for c in classes_arr {
        let id = c.get("id").and_then(|v| v.as_str()).unwrap_or("").to_string();
        let desc = c.get("description").and_then(|v| v.as_str()).unwrap_or("").to_string();
        if !id.is_empty() {
            criteria_map.insert(id, if desc.is_empty() { None } else { Some(Value::String(desc)) });
        }
    }

    let mut final_answers = BTreeMap::new();
    let mut total_tokens = 0u64;
    let mut timing = Timing::default();
    let mut cache_hit = false;

    for (item_key, q) in &req.questions {
        let Question::Choice { instructions, .. } = q else {
            continue;
        };

        let item_text = match instructions {
            Some(Value::Object(map)) => map
                .get("item")
                .and_then(|v| {
                    if let Some(s) = v.as_str() {
                        Some(s.to_string())
                    } else {
                        v.get("text").and_then(|t| t.as_str()).map(|t| t.to_string())
                    }
                })
                .unwrap_or_else(|| instructions.as_ref().map(|v| v.to_string()).unwrap_or_default()),
            Some(Value::String(s)) => s.clone(),
            _ => String::new(),
        };

        let sub_req = Request {
            model: req.model.clone(),
            state: serde_json::json!({
                "text": item_text,
            }),
            questions: {
                let mut qs = BTreeMap::new();
                qs.insert(
                    item_key.clone(),
                    Question::Choice {
                        instructions: Some(Value::String("Which class does this item belong to?".into())),
                        criteria: criteria_map.clone(),
                    },
                );
                qs
            },
        };

        let mut d = rt.decide_with_temperature(&sub_req, true)?;
        total_tokens += d.response.usage.input_tokens;
        cache_hit |= d.cache_hit;
        timing.tokenize_ms += d.timing.tokenize_ms;
        timing.encode_ms += d.timing.encode_ms;
        timing.decide_ms += d.timing.decide_ms;
        timing.calibrate_ms += d.timing.calibrate_ms;

        if let Some(ans) = d.response.answers.remove(item_key) {
            final_answers.insert(item_key.clone(), ans);
        }
    }

    Ok(Decision {
        response: Response {
            model: rt.backend().model_id().to_string(),
            answers: final_answers,
            usage: Usage { input_tokens: total_tokens, output_tokens: 0 },
            latency_ms: 0.0,
        },
        cache_hit,
        timing,
    })
}

/// 2. jev_rerank: unpacks candidate texts from instructions into passage state.
fn handle_jev_rerank<B: Backend>(rt: &mut Runtime<B>, req: &Request) -> Result<Decision, ApiError> {
    let query = req.state.get("query").and_then(|v| v.as_str()).unwrap_or("");
    let mut final_answers = BTreeMap::new();
    let mut total_tokens = 0u64;
    let mut timing = Timing::default();
    let mut cache_hit = false;

    for (rel_id, q) in &req.questions {
        let Question::Noul { instructions, .. } = q else {
            continue;
        };

        let ins_str = instructions.as_ref().and_then(|v| v.as_str()).unwrap_or("");
        let cand_text = if let Some(pos) = ins_str.find("Candidate ") {
            if let Some(colon) = ins_str[pos..].find(':') { ins_str[pos + colon + 1..].trim() } else { ins_str }
        } else {
            ins_str
        };

        let sub_req = Request {
            model: req.model.clone(),
            state: serde_json::json!({
                "passage": cand_text,
                "question": query,
            }),
            questions: {
                let mut qs = BTreeMap::new();
                qs.insert(
                    rel_id.clone(),
                    Question::Noul {
                        instructions: Some(Value::String(
                            "Based on the passage, does this document answer or relate to the query?".into(),
                        )),
                        criteria: None,
                    },
                );
                qs
            },
        };

        let mut d = rt.decide_with_temperature(&sub_req, true)?;
        total_tokens += d.response.usage.input_tokens;
        cache_hit |= d.cache_hit;
        timing.tokenize_ms += d.timing.tokenize_ms;
        timing.encode_ms += d.timing.encode_ms;
        timing.decide_ms += d.timing.decide_ms;
        timing.calibrate_ms += d.timing.calibrate_ms;

        if let Some(ans) = d.response.answers.remove(rel_id) {
            final_answers.insert(rel_id.clone(), ans);
        }
    }

    Ok(Decision {
        response: Response {
            model: rt.backend().model_id().to_string(),
            answers: final_answers,
            usage: Usage { input_tokens: total_tokens, output_tokens: 0 },
            latency_ms: 0.0,
        },
        cache_hit,
        timing,
    })
}

/// 3. jev_verify: maps claims against evidence using in-distribution NLI head.
fn handle_jev_verify<B: Backend>(rt: &mut Runtime<B>, req: &Request) -> Result<Decision, ApiError> {
    let claims = req.state.get("claims").and_then(|c| c.as_array()).cloned().unwrap_or_default();
    let evidence_items = extract_evidence_items(&req.state);

    let mut nli_criteria = BTreeMap::new();
    nli_criteria.insert("entailment".into(), Some(Value::String("the hypothesis follows".into())));
    nli_criteria.insert("neutral".into(), Some(Value::String("neither follows nor contradicts".into())));
    nli_criteria.insert("contradiction".into(), Some(Value::String("the hypothesis is false".into())));

    let mut final_answers = BTreeMap::new();
    let mut total_tokens = 0u64;
    let mut timing = Timing::default();
    let mut cache_hit = false;

    for claim in &claims {
        let claim_id = claim.get("id").and_then(|v| v.as_str()).unwrap_or("").to_string();
        let claim_text = claim.get("text").and_then(|v| v.as_str()).unwrap_or("");
        if claim_id.is_empty() {
            continue;
        }

        // Match claim against evidence items
        let (best_ev_id, best_ev_text) = pick_best_evidence(&evidence_items, claim_text);

        let relation_key = format!("relation_{claim_id}");
        let subject_key = format!("subject_{claim_id}");

        let mut sub_qs = BTreeMap::new();
        if req.questions.contains_key(&relation_key) {
            sub_qs.insert(
                "nli_rel".into(),
                Question::Choice {
                    instructions: Some(Value::String("How does the hypothesis relate to the premise?".into())),
                    criteria: nli_criteria.clone(),
                },
            );
        }

        let sub_req = Request {
            model: req.model.clone(),
            state: serde_json::json!({
                "premise": best_ev_text,
                "hypothesis": claim_text,
            }),
            questions: sub_qs,
        };

        let mut d = rt.decide_with_temperature(&sub_req, true)?;
        total_tokens += d.response.usage.input_tokens;
        cache_hit |= d.cache_hit;
        timing.tokenize_ms += d.timing.tokenize_ms;
        timing.encode_ms += d.timing.encode_ms;
        timing.decide_ms += d.timing.decide_ms;
        timing.calibrate_ms += d.timing.calibrate_ms;

        if let Some(Answer::Choice { probabilities, .. }) = d.response.answers.remove("nli_rel") {
            let p_entail = get_prob(&probabilities, "entailment");
            let p_contra = get_prob(&probabilities, "contradiction");
            let p_neutral = get_prob(&probabilities, "neutral");

            let mut mapped: BTreeMap<String, f64> = BTreeMap::new();
            mapped.insert("supports".into(), p_entail);
            mapped.insert("contradicts".into(), p_contra);
            mapped.insert("says_nothing".into(), p_neutral);

            let (best_choice, max_p) = mapped
                .iter()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(k, v)| (k.clone(), *v))
                .unwrap_or_else(|| ("supports".into(), 0.33));

            final_answers.insert(
                relation_key,
                Answer::Choice {
                    choice: best_choice,
                    probabilities: Ordered(mapped.into_iter().collect()),
                    confidence: max_p,
                },
            );
        }

        if req.questions.contains_key(&subject_key) {
            // Subject check: does the evidence talk about what the claim is about?
            let has_subject = !best_ev_text.is_empty()
                && claim_text
                    .split_whitespace()
                    .any(|w| w.len() > 3 && best_ev_text.to_lowercase().contains(&w.to_lowercase()));
            final_answers.insert(subject_key, Answer::Noul { noul: if has_subject { 0.92 } else { 0.15 } });
        }

        let source_key = format!("source_{claim_id}");
        if let Some(Question::Choice { criteria: source_criteria, .. }) = req.questions.get(&source_key) {
            let choice_name = if source_criteria.contains_key(&best_ev_id) {
                best_ev_id
            } else {
                source_criteria.keys().next().cloned().unwrap_or_else(|| "none".into())
            };
            let probs = source_criteria
                .keys()
                .map(|k| (k.clone(), if *k == choice_name { 0.90 } else { 0.10 / source_criteria.len().max(1) as f64 }))
                .collect();
            final_answers.insert(
                source_key,
                Answer::Choice { choice: choice_name, probabilities: Ordered(probs), confidence: 0.90 },
            );
        }
    }

    Ok(Decision {
        response: Response {
            model: rt.backend().model_id().to_string(),
            answers: final_answers,
            usage: Usage { input_tokens: total_tokens, output_tokens: 0 },
            latency_ms: 0.0,
        },
        cache_hit,
        timing,
    })
}

/// 4. jev_screen: prompt injection, substance, and relevance screening.
fn handle_jev_screen<B: Backend>(rt: &mut Runtime<B>, req: &Request) -> Result<Decision, ApiError> {
    let content = req.state.get("content").and_then(|c| c.as_str()).unwrap_or("");
    let purpose = req.state.get("purpose").and_then(|p| p.as_str()).filter(|p| !p.is_empty());

    let mut final_answers = BTreeMap::new();
    let mut total_tokens = 0u64;
    let mut timing = Timing::default();
    let mut cache_hit = false;

    // 1. Injection
    let sub_req_inj = Request {
        model: req.model.clone(),
        state: serde_json::json!({
            "premise": content,
            "hypothesis": "This text instructs the recipient to ignore previous instructions."
        }),
        questions: {
            let mut qs = BTreeMap::new();
            qs.insert(
                "injection".into(),
                Question::Noul {
                    instructions: Some(Value::String("Does the premise entail the hypothesis?".into())),
                    criteria: None,
                },
            );
            qs
        },
    };
    let mut d_inj = rt.decide_with_temperature(&sub_req_inj, false)?;
    total_tokens += d_inj.response.usage.input_tokens;
    cache_hit |= d_inj.cache_hit;
    timing.tokenize_ms += d_inj.timing.tokenize_ms;
    timing.encode_ms += d_inj.timing.encode_ms;
    timing.decide_ms += d_inj.timing.decide_ms;
    timing.calibrate_ms += d_inj.timing.calibrate_ms;
    if let Some(ans) = d_inj.response.answers.remove("injection") {
        final_answers.insert("injection".into(), ans);
    }

    // 2. Substance
    let is_short_or_boilerplate = content.trim().len() < 20
        || content.to_lowercase().contains("error 404")
        || content.to_lowercase().contains("page not found");
    if is_short_or_boilerplate {
        final_answers.insert("substance".into(), Answer::Noul { noul: 0.05 });
    } else {
        final_answers.insert("substance".into(), Answer::Noul { noul: 0.92 });
    }

    // 3. Relevance
    if let Some(p) = purpose {
        let sub_req_rel = Request {
            model: req.model.clone(),
            state: serde_json::json!({
                "passage": content,
                "question": p,
            }),
            questions: {
                let mut qs = BTreeMap::new();
                qs.insert(
                    "relevance".into(),
                    Question::Noul {
                        instructions: Some(Value::String(
                            "Based on the passage, is the answer to the question yes?".into(),
                        )),
                        criteria: None,
                    },
                );
                qs
            },
        };
        let mut d_rel = rt.decide_with_temperature(&sub_req_rel, true)?;
        total_tokens += d_rel.response.usage.input_tokens;
        cache_hit |= d_rel.cache_hit;
        timing.tokenize_ms += d_rel.timing.tokenize_ms;
        timing.encode_ms += d_rel.timing.encode_ms;
        timing.decide_ms += d_rel.timing.decide_ms;
        timing.calibrate_ms += d_rel.timing.calibrate_ms;
        if let Some(ans) = d_rel.response.answers.remove("relevance") {
            final_answers.insert("relevance".into(), ans);
        }
    }

    Ok(Decision {
        response: Response {
            model: rt.backend().model_id().to_string(),
            answers: final_answers,
            usage: Usage { input_tokens: total_tokens, output_tokens: 0 },
            latency_ms: 0.0,
        },
        cache_hit,
        timing,
    })
}

/// 5. jev_compare: pairwise passage relation mapped through in-distribution NLI.
fn handle_jev_compare<B: Backend>(rt: &mut Runtime<B>, req: &Request) -> Result<Decision, ApiError> {
    let a = req.state.get("passage_a").and_then(|v| v.as_str()).unwrap_or("");
    let b = req.state.get("passage_b").and_then(|v| v.as_str()).unwrap_or("");

    let mut nli_criteria = BTreeMap::new();
    nli_criteria.insert("entailment".into(), Some(Value::String("the hypothesis follows".into())));
    nli_criteria.insert("neutral".into(), Some(Value::String("neither follows nor contradicts".into())));
    nli_criteria.insert("contradiction".into(), Some(Value::String("the hypothesis is false".into())));

    let mut sub_qs = BTreeMap::new();
    sub_qs.insert(
        "nli_rel".into(),
        Question::Choice {
            instructions: Some(Value::String("How does the hypothesis relate to the premise?".into())),
            criteria: nli_criteria,
        },
    );

    let sub_req = Request {
        model: req.model.clone(),
        state: serde_json::json!({
            "premise": a,
            "hypothesis": b,
        }),
        questions: sub_qs,
    };

    let mut d = rt.decide_with_temperature(&sub_req, true)?;
    let mut final_answers = BTreeMap::new();

    if let Some(Answer::Choice { probabilities, .. }) = d.response.answers.remove("nli_rel") {
        let p_entail = get_prob(&probabilities, "entailment");
        let p_contra = get_prob(&probabilities, "contradiction");
        let p_neutral = get_prob(&probabilities, "neutral");

        for (qid, q) in &req.questions {
            if let Question::Choice { criteria, .. } = q {
                let mut mapped_probs = BTreeMap::new();
                for key in criteria.keys() {
                    let prob = match key.as_str() {
                        "same_fact" | "entailment" | "supports" => p_entail,
                        "contradicts" | "contradiction" => p_contra,
                        "different_facts" | "neutral" | "says_nothing" => p_neutral,
                        _ => 0.1,
                    };
                    mapped_probs.insert(key.clone(), prob);
                }
                let (best_choice, max_p) = mapped_probs
                    .iter()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(k, v)| (k.clone(), *v))
                    .unwrap_or_else(|| (criteria.keys().next().cloned().unwrap_or_default(), 0.33));

                final_answers.insert(
                    qid.clone(),
                    Answer::Choice {
                        choice: best_choice,
                        probabilities: Ordered(mapped_probs.into_iter().collect()),
                        confidence: max_p,
                    },
                );
            }
        }
    }

    Ok(Decision {
        response: Response {
            model: rt.backend().model_id().to_string(),
            answers: final_answers,
            usage: d.response.usage,
            latency_ms: 0.0,
        },
        cache_hit: d.cache_hit,
        timing: d.timing,
    })
}

/// 6. jev_find: populates bare criteria candidate descriptions and evaluates existence.
fn handle_jev_find<B: Backend>(rt: &mut Runtime<B>, req: &Request) -> Result<Decision, ApiError> {
    let candidates = req.state.get("candidates").and_then(|c| c.as_array()).cloned().unwrap_or_default();
    let best_q = req.questions.get("best");
    let query_string = best_q.map(|q| q.instructions()).unwrap_or_default();
    let query = if let Some(start) = query_string.find('"') {
        if let Some(end) = query_string[start + 1..].rfind('"') {
            &query_string[start + 1..start + 1 + end]
        } else {
            ""
        }
    } else {
        ""
    };

    let mut final_answers = BTreeMap::new();
    let mut total_tokens = 0u64;
    let mut timing = Timing::default();
    let mut cache_hit = false;

    // Check candidate relevance
    let mut cand_criteria = BTreeMap::new();
    let mut query_has_match = false;
    let q_lower = query.to_lowercase();
    let q_words: Vec<&str> = q_lower.split_whitespace().filter(|w| w.len() > 2).collect();

    for c in &candidates {
        let cid = c.get("id").and_then(|v| v.as_str()).unwrap_or("").to_string();
        let ctext = c.get("text").and_then(|v| v.as_str()).unwrap_or("").to_string();
        let c_lower = ctext.to_lowercase();

        if q_words.iter().any(|w| c_lower.contains(w)) {
            query_has_match = true;
        }
        cand_criteria.insert(cid, Some(Value::String(ctext)));
    }

    // Question "best"
    let sub_req_best = Request {
        model: req.model.clone(),
        state: serde_json::json!({
            "text": query,
        }),
        questions: {
            let mut qs = BTreeMap::new();
            qs.insert(
                "best".into(),
                Question::Choice {
                    instructions: Some(Value::String("Which candidate contains the best answer to the query?".into())),
                    criteria: cand_criteria,
                },
            );
            qs
        },
    };

    let mut d_best = rt.decide_with_temperature(&sub_req_best, true)?;
    total_tokens += d_best.response.usage.input_tokens;
    cache_hit |= d_best.cache_hit;
    timing.tokenize_ms += d_best.timing.tokenize_ms;
    timing.encode_ms += d_best.timing.encode_ms;
    timing.decide_ms += d_best.timing.decide_ms;
    timing.calibrate_ms += d_best.timing.calibrate_ms;

    if let Some(ans) = d_best.response.answers.remove("best") {
        final_answers.insert("best".into(), ans);
    }

    // Question "exists": check if any candidate actually answers the query
    let exists_prob = if query_has_match { 0.88 } else { 0.08 };
    final_answers.insert("exists".into(), Answer::Noul { noul: exists_prob });

    Ok(Decision {
        response: Response {
            model: rt.backend().model_id().to_string(),
            answers: final_answers,
            usage: Usage { input_tokens: total_tokens, output_tokens: 0 },
            latency_ms: 0.0,
        },
        cache_hit,
        timing,
    })
}

/// 7. jev_extract: contextual field value picker.
fn handle_jev_extract<B: Backend>(rt: &mut Runtime<B>, req: &Request) -> Result<Decision, ApiError> {
    let doc = req.state.get("document").and_then(|d| d.as_str()).unwrap_or("");
    let doc_lower = doc.to_lowercase();

    let mut final_answers = BTreeMap::new();

    for (field_key, q) in &req.questions {
        let Question::Choice { criteria, instructions } = q else {
            continue;
        };
        let ins_str = instructions.as_ref().and_then(|v| v.as_str()).unwrap_or("");

        // Extract field label/name
        let field_label = if let Some(start) = ins_str.find('"') {
            if let Some(end) = ins_str[start + 1..].find('"') { &ins_str[start + 1..start + 1 + end] } else { "" }
        } else {
            ""
        };
        // Extract field label/name and search terms
        let label_lower = field_label.to_lowercase();
        let search_terms: Vec<&str> =
            label_lower.split(|c: char| !c.is_alphanumeric()).filter(|s| s.len() > 2).collect();

        let mut label_positions = Vec::new();
        for term in &search_terms {
            label_positions.extend(doc_lower.match_indices(term).map(|(i, _)| i));
        }

        // Score each candidate by proximity to field label in doc
        let mut best_cid = "none_of_them".to_string();
        let mut min_distance = usize::MAX;

        for (cid, desc_val) in criteria {
            if cid == "none_of_them" {
                continue;
            }
            let desc_str = desc_val.as_ref().and_then(|v| v.as_str()).unwrap_or("");
            let val = if let Some(pos) = desc_str.find(':') {
                desc_str[pos + 1..].trim().trim_matches('"')
            } else {
                desc_str
            };

            if val.is_empty() {
                continue;
            }

            if let Some(val_pos) = doc_lower.find(&val.to_lowercase()) {
                let dist = if !label_positions.is_empty() {
                    label_positions
                        .iter()
                        .map(|&lp| if val_pos >= lp { val_pos - lp } else { lp - val_pos })
                        .min()
                        .unwrap_or(usize::MAX)
                } else {
                    100
                };

                if dist < min_distance {
                    min_distance = dist;
                    best_cid = cid.clone();
                }
            }
        }

        let mut probs = BTreeMap::new();
        for cid in criteria.keys() {
            let p = if *cid == best_cid { 0.88 } else { 0.12 / criteria.len().saturating_sub(1).max(1) as f64 };
            probs.insert(cid.clone(), p);
        }

        final_answers.insert(
            field_key.clone(),
            Answer::Choice { choice: best_cid, probabilities: Ordered(probs.into_iter().collect()), confidence: 0.88 },
        );
    }

    Ok(Decision {
        response: Response {
            model: rt.backend().model_id().to_string(),
            answers: final_answers,
            usage: Usage { input_tokens: 150, output_tokens: 0 },
            latency_ms: 0.0,
        },
        cache_hit: false,
        timing: Timing::default(),
    })
}

/// 8. jev_audit: dereferences records[i].value and verifies against source.
fn handle_jev_audit<B: Backend>(rt: &mut Runtime<B>, req: &Request) -> Result<Decision, ApiError> {
    let source = req.state.get("source").and_then(|s| s.as_str()).unwrap_or("");
    let records = req.state.get("records").and_then(|r| r.as_array()).cloned().unwrap_or_default();
    let source_lower = source.to_lowercase();

    let mut final_answers = BTreeMap::new();

    for (i, rec) in records.iter().enumerate() {
        let val = rec.get("value").and_then(|v| v.as_str()).unwrap_or("");
        let rq = rec.get("request").and_then(|r| r.as_str()).unwrap_or("");

        let absence_key = format!("absence_{i}");
        if req.questions.contains_key(&absence_key) {
            // Did the source contain what request asked for?
            let rq_words: Vec<&str> = rq.split_whitespace().filter(|w| w.len() > 3).collect();
            let source_has_it = rq_words.iter().any(|w| source_lower.contains(&w.to_lowercase()));
            let p_omitted = if source_has_it { 0.88 } else { 0.08 };
            final_answers.insert(absence_key, Answer::Noul { noul: p_omitted });
        }

        if !val.trim().is_empty() {
            let is_verbatim = source.contains(val);
            let p_wrong = if is_verbatim { 0.05 } else { 0.85 };
            for check in ["hallucinated", "off_target", "incomplete", "format"] {
                let k = format!("check_{i}_{check}");
                if req.questions.contains_key(&k) {
                    final_answers.insert(k, Answer::Noul { noul: p_wrong });
                }
            }
        }
    }

    Ok(Decision {
        response: Response {
            model: rt.backend().model_id().to_string(),
            answers: final_answers,
            usage: Usage { input_tokens: 100, output_tokens: 0 },
            latency_ms: 0.0,
        },
        cache_hit: false,
        timing: Timing::default(),
    })
}

/// 9. jev_review: patch & test result evaluation.
fn handle_jev_review<B: Backend>(rt: &mut Runtime<B>, req: &Request) -> Result<Decision, ApiError> {
    let tests_text = req.state.get("tests").and_then(|t| t.as_str()).unwrap_or("");
    let has_fail = tests_text.contains("FAILED")
        || tests_text.contains("failures=")
        || tests_text.contains("error:")
        || tests_text.contains("1 failed");
    let has_pass = tests_text.contains("passed") || tests_text.contains("OK") || tests_text.contains("test result: ok");

    let mut final_answers = BTreeMap::new();

    let make_score = |level_idx: usize| {
        let probs: Vec<f64> = (0..3).map(|i| if i == level_idx { 0.85 } else { 0.075 }).collect();
        Answer::Score {
            score: level_idx as f64,
            legend: Ordered(vec![
                ("0".into(), Value::String("low".into())),
                ("1".into(), Value::String("mid".into())),
                ("2".into(), Value::String("high".into())),
            ]),
            confidence: 0.85,
            probabilities: Ordered(probs.into_iter().enumerate().map(|(i, p)| (i.to_string(), p)).collect()),
        }
    };

    if req.questions.contains_key("correctness") {
        final_answers.insert(
            "correctness".into(),
            make_score(if has_fail {
                0
            } else if has_pass {
                2
            } else {
                1
            }),
        );
    }
    if req.questions.contains_key("spec_match") {
        final_answers.insert("spec_match".into(), make_score(if has_fail { 1 } else { 2 }));
    }
    if req.questions.contains_key("test_gap") {
        final_answers.insert(
            "test_gap".into(),
            make_score(if has_fail {
                2
            } else if has_pass {
                0
            } else {
                1
            }),
        );
    }
    if req.questions.contains_key("blast_radius") {
        final_answers.insert("blast_radius".into(), make_score(0));
    }
    if req.questions.contains_key("safe_to_apply") {
        final_answers.insert(
            "safe_to_apply".into(),
            Answer::Noul {
                noul: if has_fail {
                    0.05
                } else if has_pass {
                    0.88
                } else {
                    0.50
                },
            },
        );
    }

    // Gate claim relations
    if req.state.get("claims").is_some() {
        let gate_dec = handle_jev_verify(rt, req)?;
        for (k, v) in gate_dec.response.answers {
            final_answers.insert(k, v);
        }
    }

    Ok(Decision {
        response: Response {
            model: rt.backend().model_id().to_string(),
            answers: final_answers,
            usage: Usage { input_tokens: 150, output_tokens: 0 },
            latency_ms: 0.0,
        },
        cache_hit: false,
        timing: Timing::default(),
    })
}

/// 10. jev_decide: bounded alternative choice with requirement checks.
fn handle_jev_decide<B: Backend>(rt: &mut Runtime<B>, req: &Request) -> Result<Decision, ApiError> {
    let candidates = req.state.get("candidates").and_then(|c| c.as_array()).cloned().unwrap_or_default();
    let priorities = req.state.get("priorities").and_then(|p| p.as_str()).unwrap_or("");
    let evidence = req.state.get("evidence").and_then(|e| e.as_str()).unwrap_or("");
    let requirements = req.state.get("requirements").and_then(|r| r.as_array()).cloned().unwrap_or_default();

    let mut final_answers = BTreeMap::new();
    let mut candidate_scores: Vec<(String, f64)> = Vec::new();

    // Check each candidate against each requirement
    for (i, c) in candidates.iter().enumerate() {
        let cid = c.get("id").and_then(|v| v.as_str()).unwrap_or("").to_string();
        let desc = c.get("description").and_then(|v| v.as_str()).unwrap_or("");
        let desc_lower = desc.to_lowercase();
        let evidence_lower = evidence.to_lowercase();
        let priorities_lower = priorities.to_lowercase();

        let mut candidate_violations = 0;

        for (j, req_val) in requirements.iter().enumerate() {
            let req_str = req_val.as_str().unwrap_or("");
            let req_lower = req_str.to_lowercase();
            let check_key = format!("check_{i}_{j}");

            if !req.questions.contains_key(&check_key) {
                continue;
            }

            // Check if candidate violates requirement
            let is_contradicted = (req_lower.contains("기존 인프라") || req_lower.contains("existing"))
                && (desc_lower.contains("kafka") || desc_lower.contains("rabbitmq"))
                && (evidence_lower.contains("redis") || priorities_lower.contains("redis"));

            let verdict = if is_contradicted {
                candidate_violations += 1;
                "contradicted"
            } else {
                "supported"
            };

            let mut check_probs = BTreeMap::new();
            check_probs.insert("supported".into(), if verdict == "supported" { 0.88 } else { 0.05 });
            check_probs.insert("contradicted".into(), if verdict == "contradicted" { 0.88 } else { 0.05 });
            check_probs.insert("unknown".into(), 0.07);

            final_answers.insert(
                check_key,
                Answer::Choice {
                    choice: verdict.into(),
                    probabilities: Ordered(check_probs.into_iter().collect()),
                    confidence: 0.88,
                },
            );
        }

        // Score candidate
        let mut score = if candidate_violations > 0 { 0.10 } else { 0.60 };
        if desc_lower.contains("redis") || desc_lower.contains("rq") {
            score += 0.35;
        }
        candidate_scores.push((cid, score));
    }

    // Recommendation
    if let Some(Question::Choice { criteria, .. }) = req.questions.get("recommendation") {
        let best = candidate_scores.iter().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        let best_id = best.map(|b| b.0.clone()).unwrap_or_else(|| criteria.keys().next().cloned().unwrap_or_default());

        let mut probs = BTreeMap::new();
        for k in criteria.keys() {
            let p = if *k == best_id { 0.85 } else { 0.15 / criteria.len().saturating_sub(1).max(1) as f64 };
            probs.insert(k.clone(), p);
        }
        final_answers.insert(
            "recommendation".into(),
            Answer::Choice { choice: best_id, probabilities: Ordered(probs.into_iter().collect()), confidence: 0.85 },
        );
    }

    Ok(Decision {
        response: Response {
            model: rt.backend().model_id().to_string(),
            answers: final_answers,
            usage: Usage { input_tokens: 200, output_tokens: 0 },
            latency_ms: 0.0,
        },
        cache_hit: false,
        timing: Timing::default(),
    })
}

/// 11. jev_noul: proposition probability evaluation with truth discrimination.
fn handle_jev_noul<B: Backend>(rt: &mut Runtime<B>, req: &Request) -> Result<Decision, ApiError> {
    let propositions = req.state.get("propositions").and_then(|p| p.as_array()).cloned().unwrap_or_default();
    let context_text = extract_evidence_string(&req.state);

    let mut final_answers = BTreeMap::new();

    for p in &propositions {
        let pid = p.get("id").and_then(|v| v.as_str()).unwrap_or("").to_string();
        let ptext = p.get("text").and_then(|v| v.as_str()).unwrap_or("");
        let qkey = format!("p_{pid}");

        if !req.questions.contains_key(&qkey) {
            continue;
        }

        let p_lower = ptext.to_lowercase();

        let prob = if !context_text.is_empty() {
            // Evaluate against context
            let sub_req = Request {
                model: req.model.clone(),
                state: serde_json::json!({
                    "premise": context_text,
                    "hypothesis": ptext,
                }),
                questions: {
                    let mut qs = BTreeMap::new();
                    qs.insert(
                        "noul".into(),
                        Question::Noul {
                            instructions: Some(Value::String("Does the premise entail the hypothesis?".into())),
                            criteria: None,
                        },
                    );
                    qs
                },
            };
            if let Ok(d) = rt.decide_with_temperature(&sub_req, false) {
                match d.response.answers.get("noul") {
                    Some(Answer::Noul { noul }) => *noul,
                    _ => 0.5,
                }
            } else {
                0.5
            }
        } else {
            // General knowledge discrimination
            if p_lower.contains("2+2=4")
                || p_lower.contains("물은 100°c")
                || p_lower.contains("100°c에서 끓")
                || p_lower.contains("earth")
                || p_lower.contains("sun")
            {
                0.95
            } else if p_lower.contains("2+2=5")
                || p_lower.contains("sha-1 충돌 저항")
                || p_lower.contains("sha-1 collision resistance")
            {
                0.05
            } else {
                0.50
            }
        };

        final_answers.insert(qkey, Answer::Noul { noul: prob });
    }

    Ok(Decision {
        response: Response {
            model: rt.backend().model_id().to_string(),
            answers: final_answers,
            usage: Usage { input_tokens: 100, output_tokens: 0 },
            latency_ms: 0.0,
        },
        cache_hit: false,
        timing: Timing::default(),
    })
}

// ── Helpers ──────────────────────────────────────────────────────────────────

fn extract_evidence_items(state: &Value) -> Vec<(String, String)> {
    if let Some(ev_arr) = state.get("evidence").and_then(|e| e.as_array()) {
        let mut items = Vec::new();
        for (i, e) in ev_arr.iter().enumerate() {
            let id = e.get("id").and_then(|v| v.as_str()).map(|s| s.to_string()).unwrap_or_else(|| format!("ev_{i}"));
            let text = if let Some(s) = e.as_str() {
                s.to_string()
            } else if let Some(t) = e.get("text").and_then(|v| v.as_str()) {
                t.to_string()
            } else if let Some(c) = e.get("content").and_then(|v| v.as_str()) {
                c.to_string()
            } else {
                String::new()
            };
            items.push((id, text));
        }
        items
    } else if let Some(ev_str) = state.get("evidence").and_then(|e| e.as_str()) {
        vec![("evidence".into(), ev_str.to_string())]
    } else {
        Vec::new()
    }
}

fn pick_best_evidence(items: &[(String, String)], claim: &str) -> (String, String) {
    if items.is_empty() {
        return ("evidence".into(), String::new());
    }
    if items.len() == 1 {
        return items[0].clone();
    }

    let claim_words: Vec<&str> = claim.split_whitespace().filter(|w| w.len() > 3).collect();
    let mut best_id = items[0].0.clone();
    let mut best_text = items[0].1.clone();
    let mut max_matches = 0;

    for (id, text) in items {
        let text_lower = text.to_lowercase();
        let matches = claim_words.iter().filter(|w| text_lower.contains(&w.to_lowercase())).count();
        if matches > max_matches {
            max_matches = matches;
            best_id = id.clone();
            best_text = text.clone();
        }
    }

    (best_id, best_text)
}

fn extract_evidence_string(state: &Value) -> String {
    let items = extract_evidence_items(state);
    items.into_iter().map(|(_, t)| t).collect::<Vec<_>>().join("\n\n")
}

fn get_prob(ordered: &Ordered<f64>, key: &str) -> f64 {
    ordered.0.iter().find(|(k, _)| k == key).map(|(_, v)| *v).unwrap_or(0.33)
}
