"""
OpenJev Decision Engine
=======================
A high-speed, non-autoregressive System 1 decision engine and drop-in TypeSafe wire-compatible API.
Evaluates unstructured state against typed schemas (Choice, Score, Noul) in a single parallel pass.

Author: The Research Collective
License: MIT / Apache 2.0
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
from pydantic import BaseModel, Field

# Ensure UTF-8 stdout encoding for Windows compatibility
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

QTYPES = {"choice": 0, "score": 1, "noul": 2}
QTYPE_NAMES = {v: k for k, v in QTYPES.items()}


# -----------------------------------------------------------------------------
# 1. Pydantic Schemas (TypeSafe Wire Specification)
# -----------------------------------------------------------------------------

class NoulQuestion(BaseModel):
    type: Literal["noul"] = "noul"
    instructions: Union[str, Dict[str, Any], List[Any]]
    criteria: Optional[Dict[str, Optional[str]]] = None


class ChoiceQuestion(BaseModel):
    type: Literal["choice"] = "choice"
    instructions: Union[str, Dict[str, Any], List[Any]]
    criteria: Union[Dict[str, Optional[str]], List[str]]


class ScoreQuestion(BaseModel):
    type: Literal["score"] = "score"
    instructions: Union[str, Dict[str, Any], List[Any]]
    criteria: List[str]


QuestionType = Union[NoulQuestion, ChoiceQuestion, ScoreQuestion, Dict[str, Any]]


class SystemOneRequest(BaseModel):
    state: Union[str, Dict[str, Any], List[Any]]
    model: str = "openjev-latest"
    questions: Dict[str, QuestionType]


class NoulAnswer(BaseModel):
    type: Literal["noul"] = "noul"
    noul: float
    confidence: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


class ChoiceAnswer(BaseModel):
    type: Literal["choice"] = "choice"
    choice: str
    probabilities: Dict[str, float]
    confidence: float
    metadata: Optional[Dict[str, Any]] = None


class ScoreAnswer(BaseModel):
    type: Literal["score"] = "score"
    score: float
    legend: Dict[str, str]
    probabilities: Dict[str, float]
    confidence: float
    metadata: Optional[Dict[str, Any]] = None


AnswerType = Union[NoulAnswer, ChoiceAnswer, ScoreAnswer]


class UsageInfo(BaseModel):
    input_tokens: int
    output_tokens: int = 0
    total_tokens: Optional[int] = None
    latency_ms: Optional[float] = None


class SystemOneResponse(BaseModel):
    model: str
    answers: Dict[str, AnswerType]
    usage: UsageInfo


# -----------------------------------------------------------------------------
# 2. Math & Rendering Utilities
# -----------------------------------------------------------------------------

def confidence_from_probs(p: np.ndarray, k: int) -> float:
    """Calibrated confidence: 1 - normalized Shannon entropy over the candidate options."""
    if k < 2:
        return 1.0
    p_k = p[:k]
    # Clamp probabilities to avoid log(0)
    p_clamped = np.clip(p_k, 1e-12, 1.0)
    entropy = -float((p_clamped * np.log(p_clamped)).sum())
    norm_entropy = entropy / math.log(k)
    return float(np.clip(1.0 - norm_entropy, 0.0, 1.0))


def serialize_state(state: Any) -> str:
    """Serialize arbitrary Python/JSON state into a canonical string."""
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False)


def render_options(q_dict: Dict[str, Any]) -> List[str]:
    """Convert question criteria into structured label-index options."""
    qtype = q_dict["t"]
    crit = q_dict.get("crit")
    if qtype == "choice":
        if isinstance(crit, dict):
            return [k if not v else f"{k}: {v}" for k, v in crit.items()]
        elif isinstance(crit, list):
            return [str(c) for c in crit]
        return ["option_0", "option_1"]
    if qtype == "score":
        if isinstance(crit, list):
            return [f"level {i}: {c}" for i, c in enumerate(crit)]
        return ["level 0: low", "level 1: high"]
    # Noul: Always [false, true] so p[1] corresponds to P(true)
    crit = crit or {}
    false_desc = crit.get("false") if isinstance(crit, dict) else None
    true_desc = crit.get("true") if isinstance(crit, dict) else None
    return [
        "false: " + (false_desc or "no, the statement does not hold"),
        "true: " + (true_desc or "yes, the statement holds"),
    ]


def build_sequence(tok: Any, state: Any, q_dict: Dict[str, Any], max_len: int = 512, head_max_len: int = 192):
    """
    Construct the bidirectional sequence:
    [CLS] <type> question: <instructions> [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] state [SEP]
    Returns token IDs and the integer index positions of each option's [MASK] marker.
    """
    mask_tok = tok.mask_token
    mask_tok_id = tok.mask_token_id
    cls_tok_id = tok.cls_token_id or tok.bos_token_id or 0
    sep_tok_id = tok.sep_token_id or tok.eos_token_id or 2

    opts = render_options(q_dict)
    ins = str(q_dict["ins"]).replace(mask_tok, " ")
    head_text = f"{q_dict['t']} question: {ins}"
    head_ids = tok(head_text, add_special_tokens=False)["input_ids"]

    opt_ids = []
    for opt_text in opts:
        tokenized_opt = tok(" " + opt_text.replace(mask_tok, " "), add_special_tokens=False)["input_ids"][:48]
        opt_ids.append([mask_tok_id] + tokenized_opt)

    opt_budget = head_max_len - sum(len(o) for o in opt_ids)
    if opt_budget < 16:
        per = max(4, (head_max_len - 16) // max(1, len(opt_ids)))
        opt_ids = [o[:per] for o in opt_ids]
        opt_budget = head_max_len - sum(len(o) for o in opt_ids)

    head_ids = head_ids[:max(8, opt_budget)]
    ids = [cls_tok_id] + head_ids + [sep_tok_id]

    markers = []
    for o in opt_ids:
        markers.append(len(ids))
        ids.extend(o)
    ids.append(sep_tok_id)

    room = max(0, max_len - len(ids) - 1)
    state_ids = tok(serialize_state(state).replace(mask_tok, " "), add_special_tokens=False)["input_ids"][:room]
    ids = ids + state_ids + [sep_tok_id]

    valid_markers = [m for m in markers if m < max_len]
    return ids[:max_len], valid_markers


# -----------------------------------------------------------------------------
# 3. Decision Model Architecture
# -----------------------------------------------------------------------------

class DecisionScorerHead(nn.Module):
    """Transformer Decision Head that reads representations at option [MASK] markers."""

    def __init__(self, hidden_size: int, num_layers: int = 2, n_act: int = 2, dropout: float = 0.1):
        super().__init__()
        d = hidden_size
        nhead = max(1, d // 64)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=nhead, dim_feedforward=4 * d, dropout=dropout, batch_first=True, norm_first=True
        )
        self.head = nn.TransformerEncoder(encoder_layer, num_layers=num_layers) if num_layers > 0 else None
        self.type_emb = nn.Embedding(3, d)
        self.scorer = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, 1)
        )
        self.act_head = nn.Sequential(
            nn.Linear(d + 4, 256),
            nn.GELU(),
            nn.Linear(256, n_act)
        )

    def forward(self, h: torch.Tensor, attention_mask: torch.Tensor, marker_pos: torch.Tensor, marker_mask: torch.Tensor, qtype: torch.Tensor):
        # Add question type bias embedding
        h = h + self.type_emb(qtype)[:, None, :]
        if self.head is not None:
            pad_mask = ~attention_mask.bool()
            h = self.head(h, src_key_padding_mask=pad_mask)

        # Gather hidden states corresponding to the option markers
        idx = marker_pos.clamp(min=0)[:, :, None].expand(-1, -1, h.size(-1))
        m = torch.gather(h, 1, idx)
        logits = self.scorer(m).squeeze(-1).float()
        logits = logits.masked_fill(~marker_mask, -1e4)

        # Act head features: decision confidence, entropy, margin
        p = torch.softmax(logits.detach(), -1)
        k = marker_mask.sum(-1).clamp(min=2).float()
        ent = -(p * torch.log(p.clamp_min(1e-9))).sum(-1) / torch.log(k)
        top2 = p.topk(2, -1).values
        feats = torch.stack([top2[:, 0], top2[:, 0] - top2[:, 1], ent, k / 255.0], -1)
        pooled = h[:, 0].float()
        act_logits = self.act_head(torch.cat([pooled, feats], -1))

        return logits, act_logits


class OpenJevModel(nn.Module):
    """Full OpenJev Decision Model combining a bidirectional encoder with the marker scorer."""

    def __init__(self, encoder: nn.Module, head_layers: int = 2):
        super().__init__()
        self.encoder = encoder
        d = encoder.config.hidden_size
        self.decision_head = DecisionScorerHead(hidden_size=d, num_layers=head_layers)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, marker_pos: torch.Tensor, marker_mask: torch.Tensor, qtype: torch.Tensor):
        enc_out = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return self.decision_head(enc_out, attention_mask, marker_pos, marker_mask, qtype)


# -----------------------------------------------------------------------------
# 4. OpenJev High-Level Engine
# -----------------------------------------------------------------------------

class OpenJevEngine:
    """Production System One Decision Engine with parallel question fan-out."""

    def __init__(self, model_name_or_path: str = "convaiinnovations/laya", device: Optional[str] = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model_name = model_name_or_path
        self.model: Optional[nn.Module] = None
        self.tok = None
        self.cfg: Dict[str, Any] = {
            "max_len": 512,
            "head_max_len": 192,
            "temperature": [1.637, 1.251, 1.983],
            "temperature_by_options": {}
        }
        self._load_model()

    def _load_model(self):
        """Attempt loading fine-tuned weights from HF or initialize clean architecture."""
        print(f"[*] Initializing OpenJevEngine on device: {self.device}...", flush=True)
        try:
            from huggingface_hub import snapshot_download
            from transformers import AutoConfig, AutoModel, AutoTokenizer
            from safetensors.torch import load_file

            print(f"[*] Fetching weights/config for '{self.model_name}'...", flush=True)
            model_dir = snapshot_download(self.model_name)
            
            cfg_path = os.path.join(model_dir, "rl_agent_config.json")
            if os.path.exists(cfg_path):
                with open(cfg_path) as f:
                    self.cfg.update(json.load(f))

            self.tok = AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))
            enc_config = AutoConfig.from_pretrained(os.path.join(model_dir, "encoder"))
            encoder = AutoModel.from_config(enc_config)

            # Build full decision model
            self.model = OpenJevModel(encoder, head_layers=self.cfg.get("head_layers", 2))
            
            # Load safetensors weights
            weights_path = os.path.join(model_dir, "model.safetensors")
            if os.path.exists(weights_path):
                sd = load_file(weights_path)
                # Handle possible prefix naming discrepancies
                new_sd = {}
                for k, v in sd.items():
                    if k.startswith("encoder."):
                        new_sd[k] = v
                    elif k.startswith("head.") or k.startswith("type_emb.") or k.startswith("scorer.") or k.startswith("act_head."):
                        new_sd[f"decision_head.{k}"] = v
                    else:
                        new_sd[k] = v
                self.model.load_state_dict(new_sd, strict=False)
                print("[+] Pretrained OpenJev weights loaded successfully.", flush=True)

            self.model.to(self.device).eval()
        except Exception as e:
            print(f"[!] Warning: Failed loading '{self.model_name}' ({e}). Falling back to algorithmic baseline mode.", flush=True)
            self.model = None

    def _to_internal(self, qdef: Any) -> Dict[str, Any]:
        """Convert a pydantic or dict question definition to internal format."""
        if hasattr(qdef, "model_dump"):
            qdef = qdef.model_dump()
        t = qdef["type"]
        crit = qdef.get("criteria")
        if t == "choice" and isinstance(crit, list):
            crit = {c: None for c in crit}
        ins = qdef["instructions"]
        if not isinstance(ins, str):
            ins = json.dumps(ins)
        return {"t": t, "ins": ins, "crit": crit}

    @torch.no_grad()
    def system_one(self, state: Any, questions: Mapping[str, Any], model_tag: str = "openjev-latest") -> SystemOneResponse:
        """
        Evaluate unstructured state against a dictionary of typed questions in parallel.
        Returns a structured SystemOneResponse.
        """
        t0 = time.perf_counter()
        q_ids = list(questions.keys())
        if not q_ids:
            return SystemOneResponse(
                model=model_tag,
                answers={},
                usage=UsageInfo(input_tokens=0, output_tokens=0, latency_ms=0.0)
            )

        # Fallback heuristic mode if model is unavailable
        if self.model is None or self.tok is None:
            return self._heuristic_system_one(state, questions, model_tag, t0)

        items = []
        for qid in q_ids:
            q_internal = self._to_internal(questions[qid])
            seq, markers = build_sequence(
                self.tok, state, q_internal,
                max_len=self.cfg.get("max_len", 512),
                head_max_len=self.cfg.get("head_max_len", 192)
            )
            items.append({
                "ids": seq,
                "markers": markers,
                "qtype": QTYPES[q_internal["t"]],
            })

        # Collate items into a single parallel batch
        batch_size = len(items)
        max_seq_len = max(len(it["ids"]) for it in items)
        max_markers = max(len(it["markers"]) for it in items)

        pad_id = self.tok.pad_token_id or 0
        input_ids = torch.full((batch_size, max_seq_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
        marker_pos = torch.zeros((batch_size, max_markers), dtype=torch.long)
        marker_mask = torch.zeros((batch_size, max_markers), dtype=torch.bool)
        qtypes = torch.zeros((batch_size,), dtype=torch.long)

        for i, it in enumerate(items):
            seq_len = len(it["ids"])
            input_ids[i, :seq_len] = torch.tensor(it["ids"], dtype=torch.long)
            attention_mask[i, :seq_len] = 1
            n_m = len(it["markers"])
            marker_pos[i, :n_m] = torch.tensor(it["markers"], dtype=torch.long)
            marker_mask[i, :n_m] = True
            qtypes[i] = it["qtype"]

        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        marker_pos = marker_pos.to(self.device)
        marker_mask = marker_mask.to(self.device)
        qtypes = qtypes.to(self.device)

        use_amp = self.device.type == "cuda"
        amp_dtype = torch.float16 if self.device.type == "cuda" else torch.float32

        with torch.autocast(device_type=self.device.type, dtype=amp_dtype, enabled=use_amp):
            logits, act_logits = self.model(input_ids, attention_mask, marker_pos, marker_mask, qtypes)

        logits_np = logits.float().cpu().numpy()
        act_probs = torch.softmax(act_logits.float(), -1).cpu().numpy()

        answers: Dict[str, AnswerType] = {}
        temperatures = self.cfg.get("temperature", [1.637, 1.251, 1.983])

        for r, qid in enumerate(q_ids):
            q_internal = self._to_internal(questions[qid])
            k = len(items[r]["markers"])
            qt = QTYPES[q_internal["t"]]
            t_scale = temperatures[qt] if qt < len(temperatures) else 1.0

            z = logits_np[r, :k] / t_scale
            # Stable softmax
            p = np.exp(z - np.max(z))
            p = p / np.sum(p)

            meta = {"act_probability": float(act_probs[r, 0])}

            if q_internal["t"] == "choice":
                crit = q_internal["crit"]
                keys = list(crit.keys()) if isinstance(crit, dict) else [f"option_{i}" for i in range(k)]
                chosen_idx = int(np.argmax(p))
                chosen_key = keys[chosen_idx] if chosen_idx < len(keys) else keys[0]
                prob_map = {k_name: round(float(v), 4) for k_name, v in zip(keys, p)}
                answers[qid] = ChoiceAnswer(
                    type="choice",
                    choice=chosen_key,
                    probabilities=prob_map,
                    confidence=round(confidence_from_probs(p, k), 4),
                    metadata=meta
                )

            elif q_internal["t"] == "score":
                crit = q_internal["crit"] or [f"level {i}" for i in range(k)]
                legend = {str(i): str(c) for i, c in enumerate(crit)}
                prob_map = {str(i): round(float(v), 4) for i, v in enumerate(p)}
                expected_score = float(np.sum(np.arange(k) * p))
                answers[qid] = ScoreAnswer(
                    type="score",
                    score=round(expected_score, 4),
                    legend=legend,
                    probabilities=prob_map,
                    confidence=round(confidence_from_probs(p, k), 4),
                    metadata=meta
                )

            else:  # noul
                # p[1] corresponds to True
                noul_prob = float(p[1]) if k > 1 else float(p[0])
                answers[qid] = NoulAnswer(
                    type="noul",
                    noul=round(noul_prob, 4),
                    confidence=round(abs(noul_prob - 0.5) * 2.0, 4),
                    metadata=meta
                )

        latency = (time.perf_counter() - t0) * 1000.0
        n_tokens = int(attention_mask.sum().item())

        return SystemOneResponse(
            model=model_tag,
            answers=answers,
            usage=UsageInfo(input_tokens=n_tokens, output_tokens=0, latency_ms=round(latency, 2))
        )

    def _heuristic_system_one(self, state: Any, questions: Mapping[str, Any], model_tag: str, t0: float) -> SystemOneResponse:
        """Deterministic, TF-IDF / keyword similarity fallback when neural weights are absent."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        answers: Dict[str, AnswerType] = {}
        state_str = serialize_state(state)

        for qid, qdef in questions.items():
            q_internal = self._to_internal(qdef)
            opts = render_options(q_internal)
            k = len(opts)
            query = f"{state_str} {q_internal['ins']}"
            
            try:
                vec = TfidfVectorizer().fit([query] + opts)
                q_vec = vec.transform([query])
                opt_vecs = vec.transform(opts)
                sims = cosine_similarity(q_vec, opt_vecs)[0]
                z = sims * 8.0
                p = np.exp(z - np.max(z))
                p = p / np.sum(p)
            except Exception:
                p = np.ones(k) / k

            if q_internal["t"] == "choice":
                keys = list(q_internal["crit"].keys()) if isinstance(q_internal["crit"], dict) else [f"opt_{i}" for i in range(k)]
                chosen_idx = int(np.argmax(p))
                answers[qid] = ChoiceAnswer(
                    type="choice",
                    choice=keys[chosen_idx],
                    probabilities={k_name: round(float(v), 4) for k_name, v in zip(keys, p)},
                    confidence=round(confidence_from_probs(p, k), 4)
                )
            elif q_internal["t"] == "score":
                crit = q_internal["crit"] or [f"level {i}" for i in range(k)]
                answers[qid] = ScoreAnswer(
                    type="score",
                    score=round(float(np.sum(np.arange(k) * p)), 4),
                    legend={str(i): str(c) for i, c in enumerate(crit)},
                    probabilities={str(i): round(float(v), 4) for i, v in enumerate(p)},
                    confidence=round(confidence_from_probs(p, k), 4)
                )
            else:
                answers[qid] = NoulAnswer(
                    type="noul",
                    noul=round(float(p[1]), 4) if k > 1 else 0.5,
                    confidence=round(abs(p[1] - 0.5) * 2.0, 4) if k > 1 else 0.0
                )

        latency = (time.perf_counter() - t0) * 1000.0
        return SystemOneResponse(
            model=f"{model_tag}-heuristic-fallback",
            answers=answers,
            usage=UsageInfo(input_tokens=len(state_str.split()), output_tokens=0, latency_ms=round(latency, 2))
        )


# -----------------------------------------------------------------------------
# 5. FastAPI Application Server
# -----------------------------------------------------------------------------

def create_app(engine: Optional[OpenJevEngine] = None) -> Any:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware

    app = FastAPI(title="OpenJev System One API", version="1.0.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    if engine is None:
        engine = OpenJevEngine()

    @app.post("/v1/systemone", response_model=SystemOneResponse)
    async def post_system_one(request: SystemOneRequest):
        try:
            return engine.system_one(request.state, request.questions, model_tag=request.model)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/v1/models")
    async def get_models():
        return {
            "models": [
                {"name": "openjev-latest", "description": "OpenJev System One Flagship Decision Model", "release_date": "2026-09-18"},
                {"name": "jev-latest", "description": "TypeSafe Wire Compatible Alias", "release_date": "2026-09-18"},
                {"name": "laya-421m", "description": "ModernBERT Bidirectional Decision Checkpoint", "release_date": "2026-09-18"},
            ]
        }

    @app.get("/health")
    async def health_check():
        return {"status": "ok", "engine_ready": engine.model is not None, "device": str(engine.device)}

    return app


# -----------------------------------------------------------------------------
# 6. Self-Verification Smoke Test
# -----------------------------------------------------------------------------

def run_smoke_test():
    print("=" * 70, flush=True)
    print(" 🚀 OPENJEV SYSTEM ONE: VERIFICATION SMOKE TEST", flush=True)
    print("=" * 70, flush=True)

    engine = OpenJevEngine()

    sample_state = {
        "ticket": {
            "subject": "Production API Outage",
            "message": "Our payment webhook is throwing 500 errors on every checkout. Customers cannot complete purchases."
        },
        "customer_plan": "Enterprise",
        "affected_service": "Billing API"
    }

    sample_questions = {
        "is_urgent": {
            "type": "noul",
            "instructions": "Does the issue convey severe urgency or production outage?",
            "criteria": {"true": "Active downtime or blocking failure", "false": "Routine inquiry"}
        },
        "department": {
            "type": "choice",
            "instructions": "Which department should handle this ticket?",
            "criteria": {
                "infrastructure": "Server outages, network failure, 500 errors",
                "billing": "Invoice questions, card charge dispute",
                "sales": "Upgrades and contract negotiations"
            }
        },
        "severity": {
            "type": "score",
            "instructions": "How severe is the business impact?",
            "criteria": [
                "Minor: cosmetic issue with workaround",
                "Degraded: non-critical feature impaired",
                "Critical: complete revenue/checkout blockage"
            ]
        }
    }

    print("\n[*] Sending State & 3 Speculative Questions to OpenJev...", flush=True)
    response = engine.system_one(sample_state, sample_questions)

    print(f"[+] Response received from model '{response.model}' in {response.usage.latency_ms} ms:")
    print(f"    - Input tokens: {response.usage.input_tokens}, Output tokens: {response.usage.output_tokens}")
    for qid, ans in response.answers.items():
        if ans.type == "noul":
            print(f"    * [{qid.upper()}] P(true) = {ans.noul} (Confidence: {ans.confidence})")
        elif ans.type == "choice":
            print(f"    * [{qid.upper()}] Selected: '{ans.choice}' (Confidence: {ans.confidence}, Probs: {ans.probabilities})")
        elif ans.type == "score":
            print(f"    * [{qid.upper()}] Score: {ans.score:.2f} (Confidence: {ans.confidence}, Probs: {ans.probabilities})")

    print("\n[✓] Smoke test completed successfully!", flush=True)
    return response


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="OpenJev System One Engine")
    parser.add_argument("--smoke", action="store_true", help="Run self-verification smoke test")
    parser.add_argument("--serve", action="store_true", help="Serve FastAPI HTTP wire endpoint")
    parser.add_argument("--port", type=int, default=8766, help="Port to bind server on")
    args = parser.parse_args()

    if args.serve:
        import uvicorn
        app = create_app()
        print(f"[*] Serving OpenJev TypeSafe-compatible API on http://127.0.0.1:{args.port}...", flush=True)
        uvicorn.run(app, host="127.0.0.1", port=args.port)
    else:
        run_smoke_test()
