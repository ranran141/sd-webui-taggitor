import json
import base64
import mimetypes
import threading
from pathlib import Path
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from modules.script_callbacks import on_app_started, on_ui_tabs

EXTENSION_ROOT = Path(__file__).resolve().parent.parent
CONFIG_FILE = EXTENSION_ROOT / "config.json"
MODELS_DIR = EXTENSION_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)
TEMP_DIR = EXTENSION_ROOT / "tmp"
TEMP_DIR.mkdir(exist_ok=True)
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}

CAPTION_PROMPT = (
    "Describe this image concisely in 2-3 sentences. "
    "Focus on what is visually present: the subject, setting, lighting, and mood. "
    "Be direct and specific. Do not include opinions, analysis, or any text other than the description."
)

TAGGER_MODELS = {
    "wd-vit-tagger-v3": {
        "repo": "SmilingWolf/wd-vit-tagger-v3", "size": 448,
        "label": "WD ViT v3",
    },
    "wd-swinv2-tagger-v3": {
        "repo": "SmilingWolf/wd-swinv2-tagger-v3", "size": 448,
        "label": "WD SwinV2 v3",
    },
    "wd-eva02-large-tagger-v3": {
        "repo": "SmilingWolf/wd-eva02-large-tagger-v3", "size": 448,
        "label": "WD EVA02 Large v3",
    },
    "wd-v1-4-moat-tagger-v2": {
        "repo": "SmilingWolf/wd-v1-4-moat-tagger-v2", "size": 448,
        "label": "WD MOAT v2",
    },
}

_dl_status: dict = {}
_model_cache: dict = {}


# ── Config ────────────────────────────────────────────────────────────────────

def _load_config():
    try:
        if CONFIG_FILE.exists():
            return json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def _save_config(data: dict):
    cfg = _load_config()
    cfg.update(data)
    CONFIG_FILE.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")


# ── Tag IO ────────────────────────────────────────────────────────────────────

def _read_tags(img_path: Path) -> str:
    txt = img_path.with_suffix(".txt")
    return txt.read_text(encoding="utf-8").strip() if txt.exists() else ""


def _write_tags(img_path: Path, tags: str):
    txt = img_path.with_suffix(".txt")
    stripped = tags.strip()
    if not stripped:
        if txt.exists():
            txt.unlink()
        return
    txt.write_text(stripped, encoding="utf-8")


def _parse_tags(text: str) -> list:
    return [t.strip() for t in text.split(",") if t.strip()]


# ── WD Tagger ─────────────────────────────────────────────────────────────────

def _download_model(model_name: str):
    repo = TAGGER_MODELS[model_name]["repo"]
    dest = MODELS_DIR / model_name
    dest.mkdir(exist_ok=True)
    _dl_status[model_name] = {"state": "downloading", "progress": 0, "error": ""}
    try:
        from huggingface_hub import hf_hub_download
        for fname in ["model.onnx", "selected_tags.csv"]:
            hf_hub_download(repo_id=repo, filename=fname, local_dir=str(dest))
            _dl_status[model_name]["progress"] += 50
        _dl_status[model_name] = {"state": "done", "progress": 100, "error": ""}
    except Exception as e:
        _dl_status[model_name] = {"state": "error", "progress": 0, "error": str(e)}


def _model_ready(model_name: str) -> bool:
    d = MODELS_DIR / model_name
    return (d / "model.onnx").exists() and (d / "selected_tags.csv").exists()


def _load_tagger(model_name: str):
    if model_name in _model_cache:
        return _model_cache[model_name]
    import csv
    import onnxruntime as ort
    d = MODELS_DIR / model_name
    session = ort.InferenceSession(str(d / "model.onnx"),
                                   providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    tags, cats = [], []
    with open(d / "selected_tags.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            tags.append(row["name"])
            cats.append(int(row.get("category", 0)))
    _model_cache[model_name] = (session, tags, cats)
    return _model_cache[model_name]


def _preprocess(image_path: Path, size: int = 448):
    import numpy as np
    from PIL import Image
    img = Image.open(image_path).convert("RGBA")
    bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
    bg.paste(img, mask=img.split()[3])
    img = bg.convert("RGB")
    w, h = img.size
    m = max(w, h)
    sq = Image.new("RGB", (m, m), (255, 255, 255))
    sq.paste(img, ((m - w) // 2, (m - h) // 2))
    img = sq.resize((size, size), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32)
    return arr[np.newaxis]


def _run_tagger(model_name: str, image_path: Path, threshold: float = 0.35) -> list:
    session, tags, cats = _load_tagger(model_name)
    size = TAGGER_MODELS[model_name]["size"]
    arr = _preprocess(image_path, size)
    inp = session.get_inputs()[0].name
    probs = session.run(None, {inp: arr})[0][0]
    result = []
    for i, p in enumerate(probs):
        if i >= len(tags):
            break
        if cats[i] == 9:
            continue
        if p >= threshold:
            result.append((tags[i].replace("_", " "), float(p)))
    result.sort(key=lambda x: -x[1])
    return [t for t, _ in result]


# ── LM Studio helpers ─────────────────────────────────────────────────────────

def _lmstudio_base() -> str:
    return _load_config().get("lmstudio_url", "http://localhost:1234").rstrip("/")


def _get_lm_models(base: str) -> list:
    import requests
    try:
        r = requests.get(f"{base}/api/v1/models", timeout=5)
        if r.ok:
            return r.json().get("models", [])
        return []
    except Exception:
        return []


def _is_model_loaded(base: str, model_id: str) -> bool:
    for m in _get_lm_models(base):
        if m.get("key") == model_id and m.get("loaded_instances"):
            return True
    return False


def _unload_all_models(base: str) -> None:
    import requests
    try:
        for m in _get_lm_models(base):
            for inst in m.get("loaded_instances", []):
                iid = inst.get("id")
                if iid:
                    requests.post(f"{base}/api/v1/models/unload",
                                  json={"instance_id": iid}, timeout=10)
    except Exception:
        pass


def _load_lm_model(base: str, model_id: str) -> None:
    import requests
    if _is_model_loaded(base, model_id):
        return
    _unload_all_models(base)
    requests.post(f"{base}/api/v1/models/load", json={"model": model_id}, timeout=120)


# ── HTML ──────────────────────────────────────────────────────────────────────

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Taggitor</title>
<style>
:root{
  --bg:#111827;--bg2:#1f2937;--bg3:#0d1117;--bg4:#1a2234;--bg4h:#243047;
  --bd:#374151;--bd2:#4b5563;
  --txt:#f9fafb;--txt2:#d1d5db;--txt3:#9ca3af;--txt4:#6b7280;--txt5:#4b5563;
  --acc:#e2e8f0;--pri:#94a3b8;--pri-bg:rgba(148,163,184,0.12);
  --sel:#3b82f6;--sel-bg:rgba(59,130,246,0.12);
  --ok:#22c55e;--ok-bg:rgba(34,197,94,0.12);
  --danger:#ef4444;--danger-bg:rgba(239,68,68,0.12);
  --shadow:rgba(0,0,0,0.6);
}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--txt);font-family:'Segoe UI',sans-serif;
  font-size:15px;height:100vh;display:flex;flex-direction:column;overflow:hidden}

/* ── Header ── */
#hdr{background:var(--bg2);flex-shrink:0;box-shadow:0 2px 10px var(--shadow);z-index:100}
#hdr-r1{padding:0 16px;height:44px;display:flex;gap:0;align-items:center;border-bottom:1px solid var(--bd)}
.mode-toggle-wrap{display:flex;background:var(--bg3);border:1px solid var(--bd2);border-radius:9px;padding:3px;gap:2px}
.mode-tab{height:30px;padding:0 20px;background:transparent;border:none;border-radius:7px;
  font-size:13px;font-weight:600;color:var(--txt4);cursor:pointer;transition:all .15s;white-space:nowrap}
.mode-tab:hover{color:var(--txt2)}
.mode-tab.active{background:var(--bg4h);color:var(--acc);box-shadow:0 1px 4px rgba(0,0,0,.5)}
.hdr-sep{width:1px;height:22px;background:var(--bd2);flex-shrink:0;margin:0 2px}
.model-sel{height:28px;padding:0 8px;background:var(--bg3);border:1px solid var(--bd2);
  border-radius:6px;color:var(--txt);font-size:13px;cursor:pointer;width:220px}
#pv-danbooru-model,#pv-caption-model{width:220px;flex-shrink:0}
.model-sel:focus{outline:none;border-color:var(--pri)}
.thr-stepper{display:flex;align-items:center;background:var(--bg3);border:1px solid var(--bd2);
  border-radius:7px;overflow:hidden;flex-shrink:0}
.thr-stepper span{width:36px;text-align:center;font-size:13px;color:var(--txt2);
  font-variant-numeric:tabular-nums;user-select:none}
.thr-btn{width:26px;height:26px;background:transparent;border:none;color:var(--txt3);
  font-size:16px;cursor:pointer;transition:all .12s;display:flex;align-items:center;justify-content:center}
.thr-btn:hover{background:var(--bg4h);color:var(--txt)}
.analyze-btn{height:28px;padding:0 18px;background:#f97316;border:none;border-radius:6px;
  color:#fff;font-size:13px;font-weight:700;cursor:pointer;transition:all .15s;white-space:nowrap}
.analyze-btn:hover{background:#ea6c0a;box-shadow:0 0 0 3px rgba(249,115,22,.3)}
.analyze-btn:disabled{background:#7c4a1e;color:#aaa;cursor:not-allowed;box-shadow:none}
.hbtn{height:28px;width:28px;padding:0;background:transparent;border:none;
  border-radius:6px;color:var(--txt4);font-size:16px;cursor:pointer;transition:all .15s;
  display:flex;align-items:center;justify-content:center}
.hbtn:hover{background:var(--bg4);color:var(--txt)}
#pv-progress-wrap{position:absolute;bottom:0;left:0;right:0;height:2px;pointer-events:none;display:none}
#pv-progress-bar,#edit-progress-bar{height:100%;background:#f97316;transition:width .2s;width:0%}
#edit-progress-wrap{height:2px;pointer-events:none;display:none;position:absolute;bottom:0;left:0;right:0}

/* Preview header (inside right panel) — single row */
#preview-hdr{display:flex;border-bottom:1px solid var(--bd);flex-shrink:0;background:var(--bg2)}
#preview-result-pnl{flex:1;display:flex;flex-direction:column;padding:10px 12px;overflow:hidden;min-height:0}
#preview-footer{display:flex;gap:8px;justify-content:flex-end;padding:8px 12px 10px;flex-shrink:0;background:var(--bg2);border-top:1px solid var(--bd)}
#pv-controls{padding:6px 12px;display:flex;gap:8px;align-items:center;position:relative;width:100%}
.pv-toggle-wrap{display:flex;background:var(--bg3);border:1px solid var(--bd2);border-radius:8px;padding:2px;gap:2px;flex-shrink:0}
.pv-toggle-btn{height:24px;padding:0 14px;background:transparent;border:none;border-radius:6px;
  font-size:12px;font-weight:600;color:var(--txt4);cursor:pointer;transition:all .15s;white-space:nowrap}
.pv-toggle-btn:hover{color:var(--txt2)}
.pv-toggle-btn.active{background:var(--bg4h);color:var(--acc);box-shadow:0 1px 4px rgba(0,0,0,.4)}

/* ── Layout ── */
#preview-page,#edit-page{display:flex;flex:1;overflow:hidden;min-height:0}
#edit-page{display:none}
#sim-preview{display:flex;flex-direction:column;flex:4;min-width:0;
  border-right:1px solid var(--bd);overflow:hidden;position:relative;background:var(--bg);min-height:0}
#sim-drop{flex:1;border:2px dashed var(--bd2);border-radius:8px;margin:12px;
  display:flex;align-items:center;justify-content:center;flex-direction:column;gap:8px;
  color:var(--txt4);font-size:14px;cursor:pointer;transition:all .15s;text-align:center;user-select:none}
#sim-drop:hover{border-color:var(--pri);color:var(--txt3)}
#sim-drop.drag-active{border-color:var(--sel);color:#93c5fd;background:var(--sel-bg)}
#sim-img{width:100%;height:100%;flex:1;object-fit:contain;display:none;padding:8px;cursor:pointer;min-height:0}
#gp{flex:4;min-width:0;display:flex;flex-direction:column;
  border-right:1px solid var(--bd);overflow:hidden;position:relative}
#grid-empty{flex:1;display:flex;flex-direction:column;align-items:center;justify-content:center;
  gap:10px;color:var(--txt4);font-size:13px;text-align:center;padding:16px;cursor:pointer;
  user-select:none;margin:12px;border:2px dashed var(--bd2);border-radius:8px;transition:all .15s}
#grid-empty:hover{border-color:var(--pri);color:var(--txt3)}
#grid-wrap{flex:1;position:relative;overflow:hidden;display:flex;flex-direction:column}
#grid{flex:1;overflow-y:auto;padding:8px;min-height:0;
  display:grid;grid-template-columns:repeat(auto-fill,minmax(90px,1fr));
  grid-auto-rows:90px;gap:6px;align-content:start}
.tw{position:relative;overflow:hidden;cursor:pointer;background:var(--bg2);border-radius:6px}
.tw img{width:100%;height:100%;object-fit:contain;display:block}
.tw-check{display:none;position:absolute;top:5px;right:5px;width:20px;height:20px;
  background:#3b82f6;border-radius:50%;align-items:center;justify-content:center;z-index:1;pointer-events:none}
.tw.sel .tw-check{display:flex}
#drop-ov{position:absolute;inset:0;background:rgba(59,130,246,.18);
  border:3px dashed var(--sel);border-radius:4px;
  display:none;align-items:center;justify-content:center;z-index:50;pointer-events:none}
#drop-ov.show{display:flex}
#drop-ov span{font-size:17px;font-weight:700;color:#93c5fd;
  background:rgba(15,23,42,.85);padding:12px 24px;border-radius:10px}

/* ── Right panels ── */
#rp,#edit-rp{flex:6;display:flex;flex-direction:column;overflow:hidden;min-width:0}
#empty-st{flex:1;display:flex;flex-direction:column;align-items:center;justify-content:center;
  gap:10px;color:var(--txt4);font-size:20px}
#empty-st svg{opacity:.25}

/* Edit controls header — 1 row */
#edit-controls{display:flex;gap:8px;align-items:center;padding:6px 12px;position:relative;
  background:var(--bg2);border-bottom:1px solid var(--bd);flex-shrink:0}

/* Edit panel */
#single-pnl{flex:1;display:flex;flex-direction:column;overflow:hidden;min-height:0;padding:10px 12px}
.tr-input{flex:1;min-width:0;height:28px;padding:0 9px;background:var(--bg3);
  border:1px solid var(--bd2);border-radius:6px;color:var(--txt);font-size:13px}
.tr-input:focus{outline:none;border-color:var(--pri)}
.tr-btn{height:28px;padding:0 10px;background:var(--bg4);border:1px solid var(--bd2);
  border-radius:6px;color:var(--txt3);font-size:13px;cursor:pointer;transition:all .15s;white-space:nowrap;flex-shrink:0}
.tr-btn:hover{background:var(--bg4h);color:var(--txt);border-color:var(--pri)}
.te-btn{height:30px;padding:0 12px;border:1px solid var(--bd2);border-radius:7px;
  background:var(--bg4);color:var(--txt3);font-size:13px;font-weight:600;cursor:pointer;transition:all .15s}
.te-btn:hover{background:var(--bg4h);color:var(--txt);border-color:var(--pri)}
.te-btn.ok{border-color:#166534;color:#86efac}
.te-btn.ok:hover{background:var(--ok-bg);border-color:var(--ok);color:var(--ok)}
.te-btn.danger{border-color:#7f1d1d;color:#fca5a5}
.te-btn.danger:hover{background:var(--danger-bg);border-color:var(--danger);color:var(--danger)}
.te-btn.active{border-color:var(--pri);color:var(--acc);background:var(--pri-bg)}
.te-btn.flash-ok{border-color:var(--ok)!important;color:var(--ok)!important;background:var(--ok-bg)!important;transition:all .1s}
.te-btn.flash-err{border-color:var(--danger)!important;color:var(--danger)!important;background:var(--danger-bg)!important;transition:all .1s}
#chips-outer{flex:1;display:flex;flex-direction:column;overflow:hidden;min-height:0;
  background:var(--bg3);border:1px solid var(--bd2);border-radius:8px}
#chips-area{flex:1;overflow-y:auto;padding:10px;
  display:flex;flex-wrap:wrap;gap:6px;align-content:flex-start}
#chips-add-row{display:flex;align-items:center;padding:6px 10px;
  border-top:1px solid var(--bd);flex-shrink:0}
#chips-add-input{flex:1;background:transparent;border:none;outline:none;
  color:var(--txt);font-size:13px;font-family:'Segoe UI',sans-serif}
#chips-add-input::placeholder{color:var(--txt5)}
.chip{display:inline-flex;align-items:center;gap:4px;padding:4px 10px;
  background:var(--bg4);border:1px solid var(--bd);border-radius:20px;
  font-size:14px;color:var(--txt2);cursor:default;transition:all .12s;user-select:none}
.chip:hover{border-color:var(--pri);color:var(--acc);background:var(--pri-bg)}
.chip-x{color:var(--txt4);cursor:pointer;font-size:12px;line-height:1;padding:0 2px;
  border-radius:50%;transition:all .1s}
.chip-x:hover{color:var(--danger)}
.chip-cnt{font-size:10px;color:var(--txt5);padding:0 2px}

/* Preview result panel */
#preview-textarea{flex:1;resize:none;background:var(--bg3);border:1px solid var(--bd2);
  border-radius:8px;color:var(--txt);font-size:14px;font-family:'Segoe UI',sans-serif;
  padding:10px;outline:none;line-height:1.6}
#preview-textarea:focus{border-color:var(--pri)}

/* ── Edit footer ── */
#edit-footer{display:flex;gap:6px;align-items:center;
  padding:6px 12px 8px;flex-shrink:0;border-top:1px solid var(--bd);background:var(--bg2)}

/* ── Toast ── */
#toast{position:fixed;bottom:16px;right:16px;padding:9px 14px;
  background:var(--bg2);border:1px solid var(--bd2);border-radius:8px;
  font-size:13px;color:var(--txt2);box-shadow:0 4px 16px var(--shadow);
  transform:translateY(60px);opacity:0;transition:all .22s;z-index:999;pointer-events:none}
#toast.show{transform:translateY(0);opacity:1}
#toast.ok{border-color:var(--ok);color:var(--ok)}
#toast.err{border-color:var(--danger);color:var(--danger)}

/* ── Settings modal ── */
#settings-overlay{position:fixed;inset:0;background:rgba(0,0,0,.7);display:none;
  align-items:center;justify-content:center;z-index:9000}
#settings-modal{background:var(--bg2);border:1px solid var(--bd);border-radius:12px;
  width:440px;max-width:92vw;padding:24px;display:flex;flex-direction:column;gap:18px;
  box-shadow:0 12px 48px var(--shadow)}
#settings-title{display:flex;align-items:center;justify-content:space-between}
#settings-title-text{font-size:16px;font-weight:700;color:var(--acc)}
#settings-close{width:32px;height:32px;background:var(--bg4);border:none;border-radius:50%;
  color:var(--txt3);font-size:18px;cursor:pointer;display:flex;align-items:center;
  justify-content:center;transition:background .15s,color .15s;flex-shrink:0}
#settings-close:hover{background:var(--bg4h);color:var(--txt)}
.settings-row{display:flex;flex-direction:column;gap:6px}
.settings-label{font-size:14px;font-weight:600;letter-spacing:0.5px;
  text-transform:uppercase;color:var(--txt)}
.settings-input{background:var(--bg3);border:1px solid var(--bd);border-radius:8px;
  color:var(--txt);font-size:13px;padding:8px 10px;width:100%;box-sizing:border-box;
  transition:border-color .15s}
.settings-input:focus{outline:none;border-color:var(--pri)}
.settings-hint{font-size:12px;color:var(--txt4);font-weight:400;letter-spacing:0;text-transform:none}
.settings-save-btn{height:32px;padding:0 20px;background:var(--ok-bg);border:1px solid #166534;
  border-radius:8px;color:#86efac;font-size:13px;font-weight:600;cursor:pointer;
  transition:all .15s;align-self:flex-end}
.settings-save-btn:hover{background:rgba(34,197,94,.2);border-color:var(--ok);color:var(--ok)}

input[type=range]{accent-color:var(--sel)}
::-webkit-scrollbar{width:5px;height:5px}
::-webkit-scrollbar-track{background:transparent}
::-webkit-scrollbar-thumb{background:var(--bd2);border-radius:3px}
::-webkit-scrollbar-thumb:hover{background:var(--txt4)}
</style>
</head>
<body>

<div id="hdr">
  <!-- Row 1: mode tabs + settings -->
  <div id="hdr-r1">
    <div class="mode-toggle-wrap">
      <button id="tab-single" class="mode-tab active" onclick="setMode('single')">Preview</button>
      <button id="tab-dir" class="mode-tab" onclick="setMode('dir')">Edit</button>
    </div>
    <div style="flex:1"></div>
    <button class="hbtn" onclick="openSettings()" title="Settings">⚙</button>
  </div>
</div>

<!-- Preview page -->
<div id="preview-page">
  <div id="sim-preview">
    <div id="sim-drop" onclick="pvOpenFilePicker()">
      <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.2"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/><path d="m21 15-5-5L5 21"/></svg>
      <span>Click to select image<br>or drag &amp; drop here</span>
    </div>
    <img id="sim-img" onclick="pvOpenFilePicker()" title="Click to select another image"
      ondragover="event.preventDefault()" ondrop="pvHandleDrop(event)" />
  </div>

  <div id="rp">
    <!-- Preview header: single row (toggle + controls) -->
    <div id="preview-hdr">
      <div id="pv-controls">
        <div class="pv-toggle-wrap">
          <button class="pv-toggle-btn active" onclick="pvSwitchTab('dan',this)">Danbooru</button>
          <button class="pv-toggle-btn" onclick="pvSwitchTab('nat',this)">Caption</button>
        </div>
        <div class="hdr-sep"></div>
        <select id="pv-danbooru-model" class="model-sel"></select>
        <select id="pv-caption-model" class="model-sel" style="display:none"></select>
        <div class="thr-stepper" id="pv-thr-stepper" title="Threshold">
          <button class="thr-btn" onclick="pvStepThr(-1)">−</button>
          <span id="pv-thr-disp">0.35</span>
          <button class="thr-btn" onclick="pvStepThr(1)">+</button>
        </div>
        <div style="flex:1"></div>
        <button id="pv-run-btn" class="analyze-btn" onclick="pvRun()">▶ Analyze</button>
        <div id="pv-progress-wrap"><div id="pv-progress-bar"></div></div>
      </div>
    </div>

    <div id="empty-st" style="display:none">
      <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.2">
        <rect x="3" y="3" width="18" height="18" rx="2"/><path d="M3 9h18M9 21V9"/>
      </svg>
      <span>Select an image</span>
    </div>

    <!-- Preview result panel -->
    <div id="preview-result-pnl">
      <textarea id="preview-textarea" placeholder="Press Analyze to see results..."></textarea>
    </div>
    <div id="preview-footer">
      <button id="pv-copy-btn" class="te-btn" onclick="pvCopy()">Copy</button>
      <button id="pv-send-btn" class="te-btn" onclick="pvSend()">Send</button>
    </div>
  </div>
</div>

<!-- Edit page -->
<div id="edit-page">
  <div id="gp">
    <div id="grid-empty" onclick="pickDir()">
      <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.2"><path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"/></svg>
      <span>Click to select a folder</span>
    </div>
    <div id="grid-wrap" style="display:none">
      <div id="grid"></div>
      <div id="drop-ov"><span>📷 Drop images to add</span></div>
    </div>
  </div>

  <div id="edit-rp">
    <!-- Edit controls header -->
    <div id="edit-controls">
      <button class="hbtn" onclick="pickDir()" title="Open folder">
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"/></svg>
      </button>
      <button class="hbtn" onclick="sortTags()" title="Sort tags">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M3 6h18M7 12h10M11 18h2"/></svg>
      </button>
      <button class="hbtn" onclick="undoTags()" title="Undo">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M3 7v6h6"/><path d="M3 13C5 7 11 3 18 5a9 9 0 0 1 3 14"/></svg>
      </button>
      <div class="hdr-sep"></div>
      <select id="at-model-sel" class="model-sel"></select>
      <div class="thr-stepper" title="Threshold">
        <button class="thr-btn" onclick="stepAtThr(-1)">−</button>
        <span id="at-thr-disp">0.35</span>
        <button class="thr-btn" onclick="stepAtThr(1)">+</button>
      </div>
      <div style="flex:1"></div>
      <button class="analyze-btn" onclick="runAutotag()">▶ Analyze</button>
      <div id="edit-progress-wrap"><div id="edit-progress-bar"></div></div>
    </div>

    <!-- Edit panel -->
    <div id="single-pnl">
      <div id="chips-outer">
        <div id="chips-area"></div>
        <div id="chips-add-row">
          <input id="chips-add-input" placeholder="Add tags..."
            onkeydown="if(event.key==='Enter')addTagBulk()"/>
        </div>
      </div>
    </div>
    <div id="edit-footer">
      <input id="trigger-input" class="tr-input" style="width:180px;flex:none"
        placeholder="Trigger words..." onkeydown="if(event.key==='Enter')addTrigger()"/>
      <button class="tr-btn" onclick="addTrigger()">Add</button>
      <div style="flex:1"></div>
      <button class="te-btn danger" onclick="clearAllTags()">Clear All</button>
      <button id="edit-save-btn" class="te-btn ok" onclick="saveSingle()">Save</button>
    </div>
  </div>
</div>


<!-- Settings overlay -->
<div id="settings-overlay" onclick="onSettingsOverlayClick(event)">
  <div id="settings-modal">
    <div id="settings-title">
      <span id="settings-title-text">⚙ Settings</span>
      <button id="settings-close" onclick="closeSettings()">✕</button>
    </div>
    <div class="settings-row">
      <div class="settings-label">LM Studio URL</div>
      <input id="lmstudio-url-input" class="settings-input" type="text"
        placeholder="http://localhost:1234" />
    </div>
    <div class="settings-row">
      <div class="settings-label">Caption System Prompt</div>
      <textarea id="caption-prompt-input" class="settings-input"
        style="height:110px;resize:vertical;line-height:1.5;font-family:'Segoe UI',sans-serif"></textarea>
      <div class="settings-hint">Used for Caption generation in Preview mode.</div>
    </div>
    <button class="settings-save-btn" onclick="saveSettingsModal()">Save</button>
  </div>
</div>

<div id="toast"></div>

<script>
const A='/tag-editor/api';


// ════════════════════════════════════════════════════
// EDIT STATE (independent)
// ════════════════════════════════════════════════════
let images=[],selected=null,checked=new Set();
let currentTags=[],isDirty=false,modelStatus={};
let pendingTags={};
let selectedModel='';
let tagHistory=[];
let sortAlpha=false;

// ════════════════════════════════════════════════════
// PREVIEW STATE (completely independent)
// ════════════════════════════════════════════════════
const pv={selected:null};
let pvLlmModels=[], pvLlmModel='', pvDanbooruModel='';
let pvTab='dan', pvThrVal=0.35;

// ── Mode switching (DOM only, no state transfer) ──────
let mode='single';
let atThrVal=0.35;
function stepAtThr(dir){
  atThrVal=Math.round(Math.min(0.95,Math.max(0.10,atThrVal+dir*0.05))*100)/100;
  document.getElementById('at-thr-disp').textContent=atThrVal.toFixed(2);
}
function setMode(m){
  if(mode===m)return;
  mode=m;
  const isDir=m==='dir';
  document.getElementById('tab-dir').classList.toggle('active',isDir);
  document.getElementById('tab-single').classList.toggle('active',!isDir);
  document.getElementById('preview-page').style.display=isDir?'none':'flex';
  document.getElementById('edit-page').style.display=isDir?'flex':'none';
  if(isDir){
    showEditPane(selected?'single':'empty');
    renderGrid();
    _refreshView();
  }else{
    pvRestorePanel();
  }
}

// ════════════════════════════════════════════════════
// PREVIEW FUNCTIONS
// ════════════════════════════════════════════════════

function pvRestorePanel(){
  if(pv.selected){
    const img=document.getElementById('sim-img');
    img.src=`${A}/image?path=${enc(pv.selected)}`;
    img.style.display='block';
    document.getElementById('sim-drop').style.display='none';
    pvShowResult();
  }else{
    document.getElementById('sim-img').style.display='none';
    document.getElementById('sim-drop').style.display='flex';
    pvShowEmpty();
  }
}

function pvShowEmpty(){
  document.getElementById('empty-st').style.display='flex';
  document.getElementById('preview-result-pnl').style.display='none';
  document.getElementById('preview-footer').style.display='none';
}
function pvShowResult(){
  document.getElementById('empty-st').style.display='none';
  document.getElementById('preview-result-pnl').style.display='flex';
  document.getElementById('preview-footer').style.display='flex';
}

async function pvLoadImage(path){
  const img=document.getElementById('sim-img');
  img.src=`${A}/image?path=${enc(path)}`;
  img.style.display='block';
  document.getElementById('sim-drop').style.display='none';
  pv.selected=path;
  pvShowResult();
}

async function pvLoadFile(file){
  const img=document.getElementById('sim-img');
  img.src=URL.createObjectURL(file);
  img.style.display='block';
  document.getElementById('sim-drop').style.display='none';
  pv.selected=null;
  pvShowResult();
  try{
    const fd=new FormData();fd.append('file',file);
    const r=await fetch(`${A}/upload-temp`,{method:'POST',body:fd});
    const d=await r.json();
    if(d.error)return toast(d.error,'err');
    pv.selected=d.path;
  }catch(e){toast(`Error: ${e.message}`,'err');}
}

async function pvOpenFilePicker(){
  try{
    const r=await fetch(`${A}/pick-files`);
    if(!r.ok)throw new Error(`HTTP ${r.status}`);
    const d=await r.json();
    if(d.error)return toast(`Error: ${d.error}`,'err');
    if(d.files&&d.files.length>0)pvLoadImage(d.files[0]);
  }catch(e){toast(`Error: ${e.message}`,'err');}
}

function pvHandleDrop(e){
  e.preventDefault();e.stopPropagation();
  const f=e.dataTransfer.files[0];
  if(f){pvLoadFile(f);return;}
  pvOpenFilePicker();
}

function pvStepThr(dir){
  pvThrVal=Math.round(Math.min(0.95,Math.max(0.10,pvThrVal+dir*0.05))*100)/100;
  document.getElementById('pv-thr-disp').textContent=pvThrVal.toFixed(2);
}
function pvSwitchTab(tab, btn){
  pvTab=tab;
  document.querySelectorAll('.pv-toggle-btn').forEach(b=>b.classList.remove('active'));
  btn.classList.add('active');
  const isDan=tab==='dan';
  document.getElementById('pv-danbooru-model').style.display=isDan?'':'none';
  document.getElementById('pv-caption-model').style.display=isDan?'none':'';
  document.getElementById('pv-thr-stepper').style.display=isDan?'flex':'none';
}

function pvSetBusy(busy){
  document.getElementById('pv-run-btn').disabled=busy;
  const wrap=document.getElementById('pv-progress-wrap');
  const bar=document.getElementById('pv-progress-bar');
  if(busy){wrap.style.display='block';bar.style.width='50%';}
  else{bar.style.width='100%';setTimeout(()=>{wrap.style.display='none';bar.style.width='0%';},600);}
}

async function pvRun(){
  if(pvTab==='dan') await pvRunDanbooru();
  else await pvRunCaption();
}

async function pvRunDanbooru(){
  const model=document.getElementById('pv-danbooru-model').value;
  if(!model)return toast('Please select a model','err');
  if(!pv.selected)return toast('Please select an image','err');
  if(!modelStatus[model]?.ready)return toast('Model not downloaded','err');
  pvSetBusy(true);
  const thr=pvThrVal;
  try{
    const r=await fetch(`${A}/autotag`,{method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({paths:[pv.selected],models:[model],threshold:thr,merge_mode:'replace'})});
    const d=await r.json();
    if(d.error)return toast(`Error: ${d.error}`,'err');
    document.getElementById('preview-textarea').value=(d.results?.[0]?.tags||[]).join(', ');
  }catch(e){toast(`Error: ${e.message}`,'err');}
  finally{pvSetBusy(false);}
}

async function pvRunCaption(){
  const model=document.getElementById('pv-caption-model').value;
  if(!model)return toast('Please select a model','err');
  if(!pv.selected)return toast('Please select an image','err');
  pvSetBusy(true);
  try{
    const r=await fetch(`${A}/llm-caption`,{method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({path:pv.selected,model})});
    const d=await r.json();
    if(d.error)return toast(`Error: ${d.error}`,'err');
    document.getElementById('preview-textarea').value=d.caption;
  }catch(e){toast(`Error: ${e.message}`,'err');}
  finally{pvSetBusy(false);}
}

function flashBtn(btn,ok){
  const orig=btn.textContent;
  btn.style.minWidth=btn.offsetWidth+'px';
  btn.classList.add(ok?'flash-ok':'flash-err');
  btn.textContent=ok?'✓':'✕';
  setTimeout(()=>{btn.classList.remove('flash-ok','flash-err');btn.textContent=orig;btn.style.minWidth='';},1400);
}

async function pvCopy(){
  const btn=document.getElementById('pv-copy-btn');
  const text=document.getElementById('preview-textarea').value.trim();
  if(!text)return flashBtn(btn,false);
  try{await navigator.clipboard.writeText(text);flashBtn(btn,true);}
  catch(e){flashBtn(btn,false);}
}
function pvSend(){
  const btn=document.getElementById('pv-send-btn');
  const text=document.getElementById('preview-textarea').value.trim();
  if(!text)return flashBtn(btn,false);
  try{
    const par=window.parent;
    const app=typeof par.gradioApp==='function'?par.gradioApp():par.document;
    const ta=app.querySelector('#txt2img_prompt textarea');
    if(!ta){flashBtn(btn,false);return toast('txt2img prompt not found','err');}
    const existing=ta.value.trim();
    ta.value=existing?existing+'\n'+text:text;
    ta.dispatchEvent(new Event('input',{bubbles:true}));
    flashBtn(btn,true);
  }catch(e){flashBtn(btn,false);toast(`Error: ${e.message}`,'err');}
}

function pvRenderDanbooruModelSel(){
  const sel=document.getElementById('pv-danbooru-model');
  const prev=sel.value||pvDanbooruModel;
  sel.innerHTML='<option value="">-- Select --</option>';
  Object.entries(modelStatus).forEach(([name,info])=>{
    const opt=document.createElement('option');
    opt.value=name;
    opt.textContent=info.label+(info.ready?'':' (not downloaded)');
    if(!info.ready)opt.style.color='var(--txt5)';
    sel.appendChild(opt);
  });
  if(prev&&modelStatus[prev])sel.value=prev;
  pvDanbooruModel=sel.value;
  sel.onchange=async()=>{
    const name=sel.value;
    if(!name){pvDanbooruModel='';return;}
    const info=modelStatus[name];
    if(!info.ready){
      if(confirm(`"${info.label}" needs to be downloaded.\nDownload now?`)){
        await startDownload(name);
      }else{sel.value=pvDanbooruModel;return;}
    }
    pvDanbooruModel=name;
    saveConfig({pv_danbooru_model:name});
  };
}
function pvRenderCaptionModelSel(){
  const sel=document.getElementById('pv-caption-model');
  const prev=sel.value||pvLlmModel;
  sel.innerHTML='<option value="">-- Select --</option>';
  pvLlmModels.forEach(key=>{
    const opt=document.createElement('option');
    opt.value=key;opt.textContent=key.split('/').pop();
    sel.appendChild(opt);
  });
  if(prev&&pvLlmModels.includes(prev))sel.value=prev;
  pvLlmModel=sel.value;
  sel.onchange=()=>{pvLlmModel=sel.value;saveConfig({selected_llm_model:sel.value});};
}

// D&D for Preview
document.addEventListener('dragover',e=>{
  if([...e.dataTransfer.types].includes('Files')){
    e.preventDefault();
    if(mode==='single')document.getElementById('sim-drop').classList.add('drag-active');
    else document.getElementById('drop-ov').classList.add('show');
  }
});
document.addEventListener('dragleave',e=>{
  if(!e.relatedTarget||!document.body.contains(e.relatedTarget)){
    document.getElementById('drop-ov').classList.remove('show');
    document.getElementById('sim-drop').classList.remove('drag-active');
  }
});
document.addEventListener('drop',async e=>{
  e.preventDefault();
  document.getElementById('drop-ov').classList.remove('show');
  document.getElementById('sim-drop').classList.remove('drag-active');
  const paths=parsePaths(e.dataTransfer.getData('text/uri-list')||'');
  if(mode==='single'){
    if(paths.length>0){pvLoadImage(paths[0]);return;}
    if(e.dataTransfer.files.length>0){pvLoadFile(e.dataTransfer.files[0]);return;}
    return;
  }
  if(paths.length>0){await addPathsToGrid(paths);return;}
});

// ════════════════════════════════════════════════════
// EDIT FUNCTIONS
// ════════════════════════════════════════════════════

function showEditPane(p){}

async function addPathsToGrid(paths){
  const r=await fetch(`${A}/validate-paths`,{method:'POST',
    headers:{'Content-Type':'application/json'},body:JSON.stringify({paths})});
  const d=await r.json();
  let added=0;
  (d.valid||[]).forEach(p=>{
    if(images.some(i=>i.path===p))return;
    images.push({path:p,name:p.split(/[\\/]/).pop(),hasTags:false});added++;
  });
  if(added>0)renderGrid();
  else toast('No images could be added');
}

async function openFilePicker(){
  try{
    const r=await fetch(`${A}/pick-files`);
    if(!r.ok)throw new Error(`HTTP ${r.status}`);
    const d=await r.json();
    if(d.error)return toast(`Error: ${d.error}`,'err');
    if(!d.files||d.files.length===0)return;
    await addPathsToGrid(d.files);
  }catch(e){toast(`Error: ${e.message}`,'err');}
}

async function pickDir(){
  try{
    const r=await fetch(`${A}/pick-dir`);
    if(!r.ok)throw new Error(`HTTP ${r.status}`);
    const d=await r.json();
    if(d.error)return toast(`Error: ${d.error}`,'err');
    if(d.dir)await loadDir(d.dir);
  }catch(e){toast(`Error: ${e.message}`,'err');}
}
async function loadDir(dir){
  if(!dir)return toast('Please specify a folder','err');
  try{
    const r=await fetch(`${A}/images?dir=${enc(dir)}`);
    if(!r.ok)throw new Error(`HTTP ${r.status}`);
    const d=await r.json();
    if(d.error)return toast(d.error,'err');
    images=d.images;checked.clear();selected=null;
    document.getElementById('grid-empty').style.display='none';
    document.getElementById('grid-wrap').style.display='flex';
    renderGrid();showAggregateChips();
  }catch(e){toast(`Error: ${e.message}`,'err');}
}

async function addTrigger(){
  const words=parseTags(document.getElementById('trigger-input').value);
  if(!words.length)return toast('Please enter a trigger word','err');
  if(!images.length)return toast('Please open a folder','err');
  function applyWords(tags){
    const toAdd=words.filter(w=>!tags.includes(w));return[...toAdd,...tags];
  }
  _pushBulkHistory();
  await Promise.all(images.map(async img=>{
    let tags=pendingTags[img.path]!==undefined?[...pendingTags[img.path]]
      :parseTags((await fetch(`${A}/tags?path=${enc(img.path)}`).then(r=>r.json())).tags||'');
    pendingTags[img.path]=applyWords(tags);
  }));
  if(selected){currentTags=[...pendingTags[selected]];isDirty=true;renderChips();}
  else if(checked.size>=2)showCheckedAggregateChips();
  else showAggregateChips();
}

function renderGrid(){
  const g=document.getElementById('grid');g.innerHTML='';
  images.forEach(img=>{
    const w=document.createElement('div');
    w.className='tw'+(checked.has(img.path)?' sel':'');
    w.dataset.path=img.path;
    w.onclick=()=>toggleSel(img.path);
    const im=document.createElement('img');
    im.src=`${A}/image?path=${enc(img.path)}`;im.loading='lazy';im.alt=img.name;
    const ck=document.createElement('div');ck.className='tw-check';
    ck.innerHTML='<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>';
    w.appendChild(im);w.appendChild(ck);g.appendChild(w);
  });
}

async function toggleSel(path){
  if(checked.has(path))checked.delete(path);
  else checked.add(path);
  document.querySelectorAll('.tw').forEach(w=>w.classList.toggle('sel',checked.has(w.dataset.path)));
  await syncRightPanel();
}

async function syncRightPanel(){
  if(checked.size===0){
    selected=null;tagHistory=[];showAggregateChips();
  }else if(checked.size===1){
    selected=[...checked][0];tagHistory=[];
    if(pendingTags[selected]!==undefined){
      currentTags=[...pendingTags[selected]];isDirty=true;
    }else{
      const r=await fetch(`${A}/tags?path=${enc(selected)}`);const d=await r.json();
      currentTags=parseTags(d.tags||'');isDirty=false;
    }
    renderChips();
  }else{
    selected=null;showCheckedAggregateChips();
  }
}
function showCheckedAggregateChips(){
  const area=document.getElementById('chips-area');area.innerHTML='';
  const total=checked.size;const map=new Map();
  [...checked].forEach(path=>{
    (pendingTags[path]||[]).forEach(t=>map.set(t,(map.get(t)||0)+1));
  });
  const entries=[...map.entries()];
  entries.sort(sortAlpha?(a,b)=>a[0].localeCompare(b[0]):(a,b)=>b[1]-a[1]);
  entries.forEach(([tag,cnt])=>{
    const chip=document.createElement('div');chip.className='chip';
    const tagEsc=tag.replace(/\\/g,'\\\\').replace(/'/g,"\\'");
    chip.innerHTML=`<span>${esc(tag)}</span><span class="chip-cnt">${cnt}/${total}</span>`+
      `<span class="chip-x" onclick="removeTagFromChecked('${tagEsc}')">✕</span>`;
    area.appendChild(chip);
  });
}
function removeTagFromChecked(tag){
  _pushBulkHistory();
  [...checked].forEach(path=>{
    if(pendingTags[path]!==undefined)
      pendingTags[path]=pendingTags[path].filter(t=>t!==tag);
  });
  showCheckedAggregateChips();
}


function renderChips(){
  const area=document.getElementById('chips-area');area.innerHTML='';
  currentTags.forEach(tag=>{
    const chip=document.createElement('div');chip.className='chip';
    const tagEsc=tag.replace(/\\/g,'\\\\').replace(/'/g,"\\'");
    chip.innerHTML=`<span>${esc(tag)}</span><span class="chip-x" onclick="removeChip('${tagEsc}')">✕</span>`;
    area.appendChild(chip);
  });
}
function removeChip(tag){
  tagHistory.push([...currentTags]);
  currentTags=currentTags.filter(t=>t!==tag);isDirty=true;
  if(selected)pendingTags[selected]=[...currentTags];
  renderChips();
}

function sortTags(){
  sortAlpha=!sortAlpha;
  const btn=document.querySelector('#edit-controls [title="Sort tags"]');
  if(btn)btn.style.color=sortAlpha?'var(--pri)':'';
  if(checked.size===0){
    showAggregateChips();
  }else if(checked.size>=2){
    showCheckedAggregateChips();
  }else{
    tagHistory.push([...currentTags]);
    currentTags=sortAlpha?[...currentTags].sort():[...(pendingTags[selected]||currentTags)];
    isDirty=true;pendingTags[selected]=[...currentTags];renderChips();
  }
}

function undoTags(){
  if(!tagHistory.length)return;
  const h=tagHistory[tagHistory.length-1];
  if(h&&h.bulk){
    tagHistory.pop();
    Object.entries(h.snap).forEach(([k,v])=>pendingTags[k]=[...v]);
    _refreshView();
  }else if(selected){
    currentTags=tagHistory.pop();isDirty=true;
    pendingTags[selected]=[...currentTags];
    renderChips();
  }
}

function _bulkTargets(){return checked.size===0?images.map(i=>i.path):[...checked];}
function _refreshView(){
  if(selected)renderChips();
  else if(checked.size>=2)showCheckedAggregateChips();
  else showAggregateChips();
}
function _pushBulkHistory(){
  const snap={};
  Object.entries(pendingTags).forEach(([k,v])=>snap[k]=[...v]);
  tagHistory.push({bulk:true,snap});
}

async function addTagBulk(){
  const inp=document.getElementById('chips-add-input');
  const words=parseTags(inp.value);
  if(!words.length)return toast('Please enter a tag','err');
  if(checked.size===1&&selected){
    tagHistory.push([...currentTags]);
    words.forEach(w=>{if(!currentTags.includes(w))currentTags.push(w);});
    isDirty=true;pendingTags[selected]=[...currentTags];renderChips();
  }else{
    _pushBulkHistory();
    await Promise.all(_bulkTargets().map(async p=>{
      let tags=pendingTags[p]!==undefined?[...pendingTags[p]]
        :parseTags((await fetch(`${A}/tags?path=${enc(p)}`).then(r=>r.json())).tags||'');
      words.forEach(w=>{if(!tags.includes(w))tags.push(w);});pendingTags[p]=tags;
    }));
    _refreshView();
  }
  inp.value='';
}

function _updateThumbDot(path,tags){
  const img=images.find(i=>i.path===path);
  if(!img)return;img.hasTags=tags.length>0;delete pendingTags[path];
}
async function saveSingle(silent=false){
  const btn=document.getElementById('edit-save-btn');
  try{
    if(checked.size===1&&selected){
      await fetch(`${A}/save`,{method:'POST',headers:{'Content-Type':'application/json'},
        body:JSON.stringify({path:selected,tags:currentTags.join(', ')})});
      _updateThumbDot(selected,currentTags);isDirty=false;
    }else{
      const toSave=_bulkTargets().filter(p=>pendingTags[p]!==undefined);
      if(!toSave.length){if(btn)flashBtn(btn,false);return;}
      await Promise.all(toSave.map(async p=>{
        const tags=pendingTags[p];
        await fetch(`${A}/save`,{method:'POST',headers:{'Content-Type':'application/json'},
          body:JSON.stringify({path:p,tags:tags.join(', ')})});
        _updateThumbDot(p,tags);
      }));
      isDirty=false;renderGrid();
    }
    if(btn)flashBtn(btn,true);
  }catch(e){if(btn)flashBtn(btn,false);toast(`Error: ${e.message}`,'err');}
}
async function clearAllTags(){
  if(!confirm('Delete all tags?'))return;
  if(checked.size===1&&selected){
    currentTags=[];isDirty=true;pendingTags[selected]=[];renderChips();
  }else{
    _bulkTargets().forEach(p=>{pendingTags[p]=[];});
    _refreshView();
  }
}


// ── Edit: WD Tagger models ────────────────────────────
async function loadModels(){const r=await fetch(`${A}/models`);modelStatus=await r.json();}
function renderModelSel(){
  const sel=document.getElementById('at-model-sel');
  const prev=sel.value||selectedModel;
  sel.innerHTML='<option value="">-- Select --</option>';
  Object.entries(modelStatus).forEach(([name,info])=>{
    const opt=document.createElement('option');
    opt.value=name;
    opt.textContent=info.label+(info.ready?'':' (not downloaded)');
    if(!info.ready)opt.style.color='var(--txt5)';
    sel.appendChild(opt);
  });
  if(prev&&modelStatus[prev])sel.value=prev;
  selectedModel=sel.value;
  sel.onchange=async()=>{
    const name=sel.value;
    if(!name){selectedModel='';saveConfig({selected_models:[]});return;}
    const info=modelStatus[name];
    if(!info.ready){
      if(confirm(`"${info.label}" needs to be downloaded.\nDownload now?`)){
        await startDownload(name);
      }else{sel.value=selectedModel;return;}
    }
    selectedModel=name;saveConfig({selected_models:[name]});
  };
}
async function startDownload(name){
  const info=modelStatus[name];
  toast(`Downloading ${info.label}...`);
  await fetch(`${A}/download`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({model:name})});
  while(true){
    await new Promise(r=>setTimeout(r,1500));
    const r=await fetch(`${A}/dl-status?model=${name}`);const s=await r.json();
    if(s.state==='done'){
      modelStatus[name].ready=true;
      selectedModel=name;saveConfig({selected_models:[name]});
      renderModelSel();
      pvRenderDanbooruModelSel();
      document.getElementById('at-model-sel').value=name;
      break;
    }else if(s.state==='error'){toast(`Download failed: ${s.error}`,'err');break;}
  }
}

// ── Edit: autotag ─────────────────────────────────────
async function runAutotag(){
  if(!selectedModel)return toast('Please select a model','err');
  const targets=images.map(i=>i.path);
  if(!targets.length)return toast('Please open a folder','err');
  if(!modelStatus[selectedModel]?.ready)return toast('Model not downloaded','err');
  const btn=document.querySelector('#edit-controls .analyze-btn');
  const wrap=document.getElementById('edit-progress-wrap');
  const bar=document.getElementById('edit-progress-bar');
  btn.disabled=true;wrap.style.display='block';bar.style.width='0%';
  const thr=atThrVal;
  let done=0;
  for(const path of targets){
    bar.style.width=`${Math.round(done/targets.length*100)}%`;
    try{
      const r=await fetch(`${A}/autotag`,{method:'POST',headers:{'Content-Type':'application/json'},
        body:JSON.stringify({paths:[path],models:[selectedModel],threshold:thr,merge_mode:'replace'})});
      const d=await r.json();
      if(d.ok&&d.results.length>0)pendingTags[path]=d.results[0].tags;
    }catch(e){}
    done++;
  }
  bar.style.width='100%';
  setTimeout(()=>{wrap.style.display='none';bar.style.width='0%';btn.disabled=false;},1200);
  selected=null;
  showAggregateChips();
}

function showAggregateChips(){
  const area=document.getElementById('chips-area');area.innerHTML='';
  const total=images.length;
  const map=new Map();
  images.forEach(img=>{
    (pendingTags[img.path]||[]).forEach(t=>map.set(t,(map.get(t)||0)+1));
  });
  const entries=[...map.entries()];
  entries.sort(sortAlpha?(a,b)=>a[0].localeCompare(b[0]):(a,b)=>b[1]-a[1]);
  entries.forEach(([tag,cnt])=>{
    const chip=document.createElement('div');chip.className='chip';
    const tagEsc=tag.replace(/\\/g,'\\\\').replace(/'/g,"\\'");
    chip.innerHTML=`<span>${esc(tag)}</span><span class="chip-cnt">${cnt}/${total}</span>`+
      `<span class="chip-x" onclick="removeTagFromAll('${tagEsc}')">✕</span>`;
    area.appendChild(chip);
  });
}
function removeTagFromAll(tag){
  _pushBulkHistory();
  images.forEach(img=>{
    if(pendingTags[img.path]!==undefined)
      pendingTags[img.path]=pendingTags[img.path].filter(t=>t!==tag);
  });
  showAggregateChips();
}

// ── LLM models ────────────────────────────────────────
async function loadLlmModels(){
  try{
    const r=await fetch(`${A}/llm-models`);
    const d=await r.json();
    if(d.error){toast(`LM Studio: ${d.error}`,'err');return;}
    pvLlmModels=d.models||[];
    pvRenderCaptionModelSel();
  }catch(e){toast('LM Studio connection failed','err');}
}

// ── Settings ──────────────────────────────────────────
const DEFAULT_CAPTION_PROMPT="Describe this image concisely in 2-3 sentences. Focus on what is visually present: the subject, setting, lighting, and mood. Be direct and specific. Do not include opinions, analysis, or any text other than the description.";
function openSettings(){
  fetch(`${A}/config`).then(r=>r.json()).then(cfg=>{
    document.getElementById('lmstudio-url-input').value=cfg.lmstudio_url||'http://localhost:1234';
    document.getElementById('caption-prompt-input').value=cfg.caption_prompt||DEFAULT_CAPTION_PROMPT;
  });
  document.getElementById('settings-overlay').style.display='flex';
}
function closeSettings(){document.getElementById('settings-overlay').style.display='none';}
function onSettingsOverlayClick(e){if(e.target===document.getElementById('settings-overlay'))closeSettings();}
async function saveSettingsModal(){
  const url=document.getElementById('lmstudio-url-input').value.trim();
  const prompt=document.getElementById('caption-prompt-input').value.trim();
  const btn=document.querySelector('.settings-save-btn');
  await saveConfig({
    lmstudio_url:url||'http://localhost:1234',
    caption_prompt:prompt||DEFAULT_CAPTION_PROMPT,
  });
  if(btn){const o=btn.textContent;btn.textContent='✓ Saved';btn.disabled=true;
    setTimeout(()=>{btn.textContent=o;btn.disabled=false;},1000);}
  closeSettings();
}

// ── Utilities ─────────────────────────────────────────
function parsePaths(uriList){
  return uriList.split(/\r?\n/).map(u=>u.trim())
    .filter(u=>u.startsWith('file:'))
    .map(u=>decodeURIComponent(u.replace(/^file:\/\/\//,'').replace(/^file:\/\//,'')).replace(/\//g,'\\'));
}
function parseTags(s){return s.split(',').map(t=>t.trim()).filter(Boolean);}
function esc(s){return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');}
function enc(s){return encodeURIComponent(s);}
async function saveConfig(d){await fetch(`${A}/config`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(d)});}
let toastT;
function toast(msg,type=''){
  const el=document.getElementById('toast');el.textContent=msg;
  el.className='show'+(type?' '+type:'');
  clearTimeout(toastT);toastT=setTimeout(()=>el.className='',2800);
}

// ── Init ──────────────────────────────────────────────
(async()=>{
  try{
    await Promise.all([loadModels(),loadLlmModels()]);
    const r=await fetch(`${A}/config`);
    if(!r.ok)throw new Error(`API ${r.status}`);
    const cfg=await r.json();
    // Edit
    if(cfg.selected_models?.[0]&&modelStatus[cfg.selected_models[0]])
      selectedModel=cfg.selected_models[0];
    renderModelSel();
    pvRestorePanel();
    // Preview
    if(cfg.pv_danbooru_model)pvDanbooruModel=cfg.pv_danbooru_model;
    if(cfg.selected_llm_model)pvLlmModel=cfg.selected_llm_model;
    pvRenderDanbooruModelSel();
    pvRenderCaptionModelSel();
  }catch(e){
    toast(`API connection error: ${e.message}`,'err');
  }
})();
</script>
</body>
</html>"""


# ── Routes ────────────────────────────────────────────────────────────────────

def register_routes(app: FastAPI):

    @app.get("/tag-editor/", response_class=HTMLResponse)
    async def page():
        return HTML_PAGE

    @app.get("/tag-editor/api/config")
    async def get_cfg():
        return JSONResponse(_load_config())

    @app.post("/tag-editor/api/config")
    async def post_cfg(req: Request):
        _save_config(await req.json()); return JSONResponse({"ok": True})

    @app.get("/tag-editor/api/pick-dir")
    async def pick_dir():
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk(); root.withdraw()
            root.wm_attributes("-topmost", 1)
            folder = filedialog.askdirectory(title="Select Image Folder")
            root.destroy()
            return JSONResponse({"dir": folder or ""})
        except Exception as e:
            return JSONResponse({"dir": "", "error": str(e)})

    @app.post("/tag-editor/api/upload-temp")
    async def upload_temp(file: UploadFile = File(...)):
        if Path(file.filename).suffix.lower() not in IMAGE_EXTS:
            return JSONResponse({"error": "Unsupported file format"})
        dest = TEMP_DIR / Path(file.filename).name
        dest.write_bytes(await file.read())
        return JSONResponse({"path": str(dest)})

    @app.post("/tag-editor/api/validate-paths")
    async def validate_paths(req: Request):
        data = await req.json()
        valid = [p for p in data.get("paths", [])
                 if Path(p).exists() and Path(p).suffix.lower() in IMAGE_EXTS]
        return JSONResponse({"valid": valid})

    @app.get("/tag-editor/api/pick-files")
    async def pick_files():
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk(); root.withdraw()
            root.wm_attributes("-topmost", 1)
            files = filedialog.askopenfilenames(
                title="Select Image Files",
                filetypes=[("Image files", "*.png *.jpg *.jpeg *.webp *.gif *.bmp"), ("All files", "*.*")]
            )
            root.destroy()
            return JSONResponse({"files": list(files)})
        except Exception as e:
            return JSONResponse({"files": [], "error": str(e)})

    @app.get("/tag-editor/api/images")
    async def list_images(dir: str = ""):
        p = Path(dir)
        if not p.exists() or not p.is_dir():
            return JSONResponse({"error": "Folder not found"})
        imgs = [{"path": str(f), "name": f.name, "hasTags": f.with_suffix(".txt").exists()}
                for f in sorted(p.iterdir()) if f.suffix.lower() in IMAGE_EXTS]
        return JSONResponse({"images": imgs})

    @app.get("/tag-editor/api/image")
    async def serve_image(path: str = ""):
        p = Path(path)
        if not p.exists(): return Response(status_code=404)
        return FileResponse(str(p), media_type=mimetypes.guess_type(str(p))[0] or "image/png")

    @app.get("/tag-editor/api/tags")
    async def get_tags(path: str = ""):
        p = Path(path)
        if not p.exists(): return JSONResponse({"error": "File not found"})
        return JSONResponse({"tags": _read_tags(p)})

    @app.post("/tag-editor/api/save")
    async def save_tags(req: Request):
        data = await req.json()
        p = Path(data.get("path", ""))
        if not p.exists(): return JSONResponse({"error": "File not found"})
        _write_tags(p, data.get("tags", ""))
        return JSONResponse({"ok": True})

    @app.get("/tag-editor/api/models")
    async def get_models():
        result = {}
        for name, info in TAGGER_MODELS.items():
            result[name] = {"repo": info["repo"], "label": info["label"], "ready": _model_ready(name)}
        return JSONResponse(result)

    @app.post("/tag-editor/api/download")
    async def start_download(req: Request):
        data = await req.json()
        name = data.get("model", "")
        if name not in TAGGER_MODELS:
            return JSONResponse({"error": "Unknown model"})
        if _dl_status.get(name, {}).get("state") == "downloading":
            return JSONResponse({"ok": True, "state": "already"})
        t = threading.Thread(target=_download_model, args=(name,), daemon=True)
        t.start()
        return JSONResponse({"ok": True})

    @app.get("/tag-editor/api/dl-status")
    async def dl_status(model: str = ""):
        return JSONResponse(_dl_status.get(model, {"state": "idle", "progress": 0, "error": ""}))

    @app.post("/tag-editor/api/autotag")
    async def autotag(req: Request):
        data = await req.json()
        paths = data.get("paths", [])
        models = data.get("models", [])
        threshold = float(data.get("threshold", 0.35))
        for m in models:
            if not _model_ready(m):
                return JSONResponse({"error": f"Model not downloaded: {m}"})
        try:
            results = []
            for path_str in paths:
                p = Path(path_str)
                if not p.exists(): continue
                seen = []; seen_set = set()
                for m in models:
                    for tag in _run_tagger(m, p, threshold):
                        if tag not in seen_set:
                            seen.append(tag); seen_set.add(tag)
                results.append({"path": path_str, "tags": seen})
            return JSONResponse({"ok": True, "count": len(results), "results": results})
        except Exception as e:
            return JSONResponse({"error": str(e)})

    @app.get("/tag-editor/api/llm-models")
    async def llm_models():
        base = _lmstudio_base()
        try:
            keys = [m["key"] for m in _get_lm_models(base) if m.get("key")]
            return JSONResponse({"models": keys})
        except Exception as e:
            return JSONResponse({"models": [], "error": str(e)})

    @app.post("/tag-editor/api/llm-caption")
    async def llm_caption(req: Request):
        import requests as req_lib
        data = await req.json()
        path_str = data.get("path", "")
        model_key = data.get("model", "")
        base = _lmstudio_base()
        p = Path(path_str)
        if not p.exists():
            return JSONResponse({"error": "File not found"})
        try:
            if model_key:
                _load_lm_model(base, model_key)
            suffix = p.suffix.lower().lstrip(".")
            mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png",
                    "webp": "image/webp", "gif": "image/gif", "bmp": "image/bmp"}.get(suffix, "image/png")
            data_url = f"data:{mime};base64,{base64.b64encode(p.read_bytes()).decode()}"
            prompt = _load_config().get("caption_prompt") or CAPTION_PROMPT
            payload = {
                "model": model_key or "local-model",
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ]}],
                "max_tokens": 500, "temperature": 0.3, "stream": False,
            }
            r = req_lib.post(f"{base}/v1/chat/completions", json=payload, timeout=120)
            caption = r.json()["choices"][0]["message"]["content"]
            return JSONResponse({"ok": True, "caption": caption})
        except Exception as e:
            return JSONResponse({"error": str(e)})


# ── Gradio tab ────────────────────────────────────────────────────────────────

def create_ui():
    import gradio as gr
    with gr.Blocks() as ui:
        gr.HTML('''
        <iframe src="/tag-editor/"
          style="width:100%;height:82vh;border:none;border-radius:8px;display:block"></iframe>
        ''')
    return [(ui, "Taggitor", "tag_editor")]


def on_started(_: None, app: FastAPI):
    register_routes(app)


on_app_started(on_started)
on_ui_tabs(create_ui)
