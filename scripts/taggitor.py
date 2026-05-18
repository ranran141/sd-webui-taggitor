import json
import mimetypes
import threading
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from modules.script_callbacks import on_app_started, on_ui_tabs

EXTENSION_ROOT = Path(__file__).resolve().parent.parent
CONFIG_FILE = EXTENSION_ROOT / "config.json"
MODELS_DIR = EXTENSION_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}

TAGGER_MODELS = {
    "wd-vit-tagger-v3": {
        "repo": "SmilingWolf/wd-vit-tagger-v3", "size": 448,
        "label": "WD ViT v3", "desc": "汎用・バランス型。速度と精度のバランスが良い",
    },
    "wd-swinv2-tagger-v3": {
        "repo": "SmilingWolf/wd-swinv2-tagger-v3", "size": 448,
        "label": "WD SwinV2 v3", "desc": "高精度。v3世代の中で安定した検出率",
    },
    "wd-eva02-large-tagger-v3": {
        "repo": "SmilingWolf/wd-eva02-large-tagger-v3", "size": 448,
        "label": "WD EVA02 Large v3", "desc": "最高精度・低速。精度重視の最終確認向け",
    },
    "wd-v1-4-moat-tagger-v2": {
        "repo": "SmilingWolf/wd-v1-4-moat-tagger-v2", "size": 448,
        "label": "WD MOAT v2", "desc": "軽量・高速。旧世代だが動作が安定",
    },
}

_dl_status: dict = {}   # model_name -> {state, progress, error}
_model_cache: dict = {} # model_name -> (session, tags_list, tag_categories)


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


# ── Model management ──────────────────────────────────────────────────────────

def _download_model(model_name: str):
    repo = TAGGER_MODELS[model_name]["repo"]
    dest = MODELS_DIR / model_name
    dest.mkdir(exist_ok=True)
    _dl_status[model_name] = {"state": "downloading", "progress": 0, "error": ""}
    try:
        from huggingface_hub import hf_hub_download
        for fname in ["model.onnx", "selected_tags.csv"]:
            _dl_status[model_name]["progress"] += 0
            hf_hub_download(repo_id=repo, filename=fname, local_dir=str(dest))
            _dl_status[model_name]["progress"] += 50
        _dl_status[model_name] = {"state": "done", "progress": 100, "error": ""}
    except Exception as e:
        _dl_status[model_name] = {"state": "error", "progress": 0, "error": str(e)}


def _model_ready(model_name: str) -> bool:
    d = MODELS_DIR / model_name
    return (d / "model.onnx").exists() and (d / "selected_tags.csv").exists()


def _load_model(model_name: str):
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
    return arr[np.newaxis]  # (1, H, W, C)


def _run_tagger(model_name: str, image_path: Path, threshold: float = 0.35) -> list:
    session, tags, cats = _load_model(model_name)
    size = TAGGER_MODELS[model_name]["size"]
    arr = _preprocess(image_path, size)
    inp = session.get_inputs()[0].name
    probs = session.run(None, {inp: arr})[0][0]
    result = []
    for i, p in enumerate(probs):
        if i >= len(tags):
            break
        if cats[i] == 9:  # skip rating tags
            continue
        if p >= threshold:
            result.append((tags[i].replace("_", " "), float(p)))
    result.sort(key=lambda x: -x[1])
    return [t for t, _ in result]


# ── HTML ──────────────────────────────────────────────────────────────────────

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<title>Taggitor</title>
<style>
:root{
  --bg:#111827;--bg2:#1f2937;--bg3:#0d1117;--bg4:#1a2234;--bg4h:#243047;
  --bd:#374151;--bd2:#4b5563;
  --txt:#f9fafb;--txt2:#d1d5db;--txt3:#9ca3af;--txt4:#6b7280;--txt5:#4b5563;
  --acc:#e2e8f0;--pri:#94a3b8;--pri-bg:rgba(148,163,184,0.12);--pri-bg2:rgba(148,163,184,0.22);
  --sel:#3b82f6;--sel-bg:rgba(59,130,246,0.12);
  --ok:#22c55e;--ok-bg:rgba(34,197,94,0.12);
  --danger:#ef4444;--danger-bg:rgba(239,68,68,0.12);
  --shadow:rgba(0,0,0,0.6);
}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--txt);font-family:'Segoe UI',sans-serif;
  font-size:15px;height:100vh;display:flex;flex-direction:column;overflow:hidden}
#hdr{background:var(--bg2);flex-shrink:0;box-shadow:0 2px 10px var(--shadow);z-index:100;
  display:flex;flex-direction:column}
#hdr-r1{padding:6px 16px;display:flex;gap:8px;align-items:center;flex-wrap:wrap;position:relative}
.mode-tab{height:32px;padding:0 14px;background:var(--bg4);border:1px solid var(--bd2);
  font-size:13px;font-weight:600;color:var(--txt4);cursor:pointer;transition:all .15s;white-space:nowrap}
.mode-tab:first-child{border-radius:7px 0 0 7px}
.mode-tab:last-child{border-radius:0 7px 7px 0;border-left:none}
.mode-tab.active{background:var(--pri-bg);border-color:var(--pri);color:var(--acc)}
#dir-input{height:28px;padding:0 9px;
  background:var(--bg3);border:1px solid var(--bd2);border-radius:6px;color:var(--txt);font-size:14px}
#dir-input:focus{outline:none;border-color:var(--pri)}
#dnd-zone{flex:1;height:34px;border:2px dashed var(--bd2);border-radius:6px;
  display:flex;align-items:center;justify-content:center;gap:8px;
  color:var(--txt4);font-size:13px;cursor:pointer;transition:all .15s;padding:0 12px;user-select:none}
#dnd-zone:hover{border-color:var(--pri);color:var(--txt3)}
#dnd-zone.drag-active{border-color:var(--sel);color:#93c5fd;background:var(--sel-bg)}

.hdr-sep{width:1px;height:22px;background:var(--bd2);flex-shrink:0;margin:0 4px}
.tr-label{font-size:13px;color:var(--txt3);white-space:nowrap;font-weight:600}
.tr-input{width:200px;height:28px;padding:0 9px;background:var(--bg3);
  border:1px solid var(--bd2);border-radius:6px;color:var(--txt);font-size:14px}
.tr-input:focus{outline:none;border-color:var(--pri)}
.tr-btn{height:28px;padding:0 10px;background:var(--bg4);border:1px solid var(--bd2);
  border-radius:6px;color:var(--txt3);font-size:13px;cursor:pointer;transition:all .15s;white-space:nowrap}
.tr-btn:hover{background:var(--bg4h);color:var(--txt);border-color:var(--pri)}
#at-model-sel{height:28px;padding:0 8px;background:var(--bg3);border:1px solid var(--bd2);
  border-radius:6px;color:var(--txt);font-size:13px;cursor:pointer;max-width:200px}
#at-model-sel:focus{outline:none;border-color:var(--pri)}
#at-thr{width:80px}
#at-thr-disp{width:32px;font-size:13px;color:var(--txt3);text-align:center}
#at-run{height:32px;padding:0 20px;background:#f97316;border:none;
  border-radius:7px;color:#fff;font-size:14px;font-weight:700;
  cursor:pointer;transition:all .15s;white-space:nowrap;flex-shrink:0;letter-spacing:.03em;min-width:80px}
#at-run:hover{background:#ea6c0a;box-shadow:0 0 0 3px rgba(249,115,22,.3)}
#at-run:disabled{background:#7c4a1e;color:#aaa;cursor:not-allowed;box-shadow:none}
#at-progress-wrap{position:absolute;bottom:0;left:0;right:0;height:3px;pointer-events:none;display:none}
#at-progress-bar{height:100%;background:#f97316;transition:width .2s;width:0%}
.hbtn{height:34px;padding:0 12px;background:var(--bg4);border:1px solid var(--bd2);
  border-radius:8px;color:var(--txt3);font-size:14px;font-weight:600;
  cursor:pointer;transition:all .15s;white-space:nowrap}
.hbtn:hover{background:var(--bg4h);color:var(--txt)}
.hbtn.active{border-color:var(--pri);color:var(--acc);background:var(--pri-bg)}
#count{font-size:13px;color:var(--txt4);white-space:nowrap}
#main{display:flex;flex:1;overflow:hidden}
#sim-preview{display:none;flex-direction:column;flex:2;min-width:0;
  border-right:1px solid var(--bd);overflow:hidden;position:relative;background:var(--bg)}
#sim-toolbar{padding:7px 10px;display:flex;gap:6px;align-items:center;
  border-bottom:1px solid var(--bd);background:var(--bg2);flex-shrink:0;font-size:13px}
#sim-filename{flex:1;color:var(--txt4);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
#sim-drop{flex:1;border:2px dashed var(--bd2);border-radius:8px;margin:16px;
  display:flex;align-items:center;justify-content:center;flex-direction:column;gap:8px;
  color:var(--txt4);font-size:14px;cursor:pointer;transition:all .15s;
  text-align:center;user-select:none}
#sim-drop:hover{border-color:var(--pri);color:var(--txt3)}
#sim-drop.drag-active{border-color:var(--sel);color:#93c5fd;background:var(--sel-bg)}
#sim-img{width:100%;flex:1;object-fit:contain;display:none;padding:8px;box-sizing:border-box;cursor:pointer}
#gp{flex:2;min-width:0;display:flex;flex-direction:column;
  border-right:1px solid var(--bd);overflow:hidden;position:relative}
#gtb{padding:7px 10px;display:flex;gap:6px;align-items:center;
  border-bottom:1px solid var(--bd);background:var(--bg2);flex-shrink:0;font-size:13px}
#gtb label{color:var(--txt3);display:flex;align-items:center;gap:4px;cursor:pointer}
#sel-count{flex:1;text-align:right;color:var(--txt4)}
#grid-wrap{flex:1;position:relative;overflow:hidden;display:flex;flex-direction:column}
#grid{flex:1;overflow-y:auto;padding:8px;min-height:0;
  display:grid;grid-template-columns:repeat(auto-fill,minmax(84px,1fr));gap:6px;
  align-content:start}
.tw{position:relative;aspect-ratio:1;border-radius:6px;overflow:hidden;
  cursor:pointer;border:2px solid transparent;transition:border-color .12s;background:var(--bg3)}
.tw:hover{border-color:var(--pri)}
.tw.sel{border-color:var(--sel)}
.tw.chk{outline:2px solid var(--ok);outline-offset:-2px}
.tw img{width:100%;height:100%;object-fit:contain;display:block}
.tchk{position:absolute;top:3px;left:3px;width:15px;height:15px;accent-color:var(--ok);cursor:pointer}
#drop-ov{position:absolute;inset:0;background:rgba(59,130,246,.18);
  border:3px dashed var(--sel);border-radius:4px;
  display:none;align-items:center;justify-content:center;z-index:50;pointer-events:none}
#drop-ov.show{display:flex}
#drop-ov span{font-size:17px;font-weight:700;color:#93c5fd;
  background:rgba(15,23,42,.85);padding:12px 24px;border-radius:10px}
#rp{flex:3;display:flex;flex-direction:column;overflow:hidden;min-width:0}
#empty-st{flex:1;display:flex;flex-direction:column;align-items:center;justify-content:center;
  gap:10px;color:var(--txt4);font-size:20px}
#empty-st svg{opacity:.25}
#single-pnl{flex:1;display:flex;flex-direction:column;padding:14px 16px;gap:10px;overflow:hidden;min-height:0}
#single-hdr{display:flex;align-items:center;gap:8px;flex-shrink:0}
.te-btn{height:30px;padding:0 12px;border:1px solid var(--bd2);border-radius:7px;
  background:var(--bg4);color:var(--txt3);font-size:13px;font-weight:600;cursor:pointer;transition:all .15s}
.te-btn:hover{background:var(--bg4h);color:var(--txt);border-color:var(--pri)}
.te-btn.ok{border-color:#166534;color:#86efac}
.te-btn.ok:hover{background:var(--ok-bg);border-color:var(--ok);color:var(--ok)}
.te-btn.danger{border-color:#7f1d1d;color:#fca5a5}
.te-btn.danger:hover{background:var(--danger-bg);border-color:var(--danger);color:var(--danger)}
.te-btn.active{border-color:var(--pri);color:var(--acc);background:var(--pri-bg)}
#chips-outer{flex:1;display:flex;flex-direction:column;gap:0;overflow:hidden;min-height:0}
#tag-textarea{flex:1;resize:none;background:var(--bg3);border:1px solid var(--bd2);border-radius:8px;
  color:var(--txt);font-size:14px;font-family:'Segoe UI',sans-serif;padding:10px;outline:none;
  line-height:1.6;display:none}
#tag-textarea:focus{border-color:var(--pri)}
#chips-area{flex:1;overflow-y:auto;padding:10px;
  background:var(--bg3);border:1px solid var(--bd2);border-radius:8px 8px 0 0;
  display:flex;flex-wrap:wrap;gap:6px;align-content:flex-start;cursor:text}
#chips-area:focus-within{border-color:var(--pri)}
.chip{display:inline-flex;align-items:center;gap:4px;padding:4px 10px;
  background:var(--bg4);border:1px solid var(--bd);border-radius:20px;
  font-size:14px;color:var(--txt2);cursor:default;transition:all .12s;user-select:none}
.chip:hover{border-color:var(--pri);color:var(--acc);background:var(--pri-bg)}
.chip-x{color:var(--txt4);cursor:pointer;font-size:12px;line-height:1;padding:0 2px;
  border-radius:50%;transition:all .1s}
.chip-x:hover{color:var(--danger)}
.chip-cnt{font-size:10px;color:var(--txt5);padding:0 2px}
#tag-input-wrap{background:var(--bg3);border:1px solid var(--bd2);border-top:none;
  border-radius:0 0 8px 8px;padding:7px 12px;display:flex;align-items:center;gap:8px;flex-shrink:0}
#tag-input-wrap:focus-within{border-color:var(--pri)}
#tag-add-input{flex:1;background:transparent;border:none;outline:none;
  color:var(--txt);font-size:14px;font-family:'Segoe UI',sans-serif}
#tag-add-input::placeholder{color:var(--txt5)}
.inp{flex:1;min-width:100px;padding:6px 10px;background:var(--bg3);
  border:1px solid var(--bd2);border-radius:6px;color:var(--txt);font-size:14px}
.inp:focus{outline:none;border-color:var(--pri)}
.mc-badge{display:inline-block;padding:2px 8px;border-radius:10px;font-size:11px;font-weight:600}
.mc-badge.ready{background:var(--ok-bg);color:var(--ok);border:1px solid #166534}
.mc-badge.missing{background:var(--danger-bg);color:#fca5a5;border:1px solid #7f1d1d}
.mc-badge.dl{background:rgba(59,130,246,.12);color:#93c5fd;border:1px solid #1d4ed8}
input[type=range]{accent-color:var(--sel)}
::-webkit-scrollbar{width:5px;height:5px}
::-webkit-scrollbar-track{background:transparent}
::-webkit-scrollbar-thumb{background:var(--bd2);border-radius:3px}
::-webkit-scrollbar-thumb:hover{background:var(--txt4)}
#toast{position:fixed;bottom:16px;right:16px;padding:9px 14px;
  background:var(--bg2);border:1px solid var(--bd2);border-radius:8px;
  font-size:13px;color:var(--txt2);box-shadow:0 4px 16px var(--shadow);
  transform:translateY(60px);opacity:0;transition:all .22s;z-index:999;pointer-events:none}
#toast.show{transform:translateY(0);opacity:1}
#toast.ok{border-color:var(--ok);color:var(--ok)}
#toast.err{border-color:var(--danger);color:var(--danger)}
</style>
</head>
<body>

<div id="hdr">
  <div id="hdr-r1">
    <div style="display:flex">
      <button id="tab-single" class="mode-tab" onclick="setMode('single')">単体画像</button>
      <button id="tab-dir" class="mode-tab active" onclick="setMode('dir')">フォルダ</button>
    </div>
    <div id="dir-row" style="display:flex;gap:6px;align-items:center;margin-left:8px">
      <input id="dir-input" type="text" placeholder="フォルダパスを入力して Enter..."
        style="width:240px" onkeydown="if(event.key==='Enter')loadDir()" />
      <button class="hbtn" onclick="pickDir()" style="padding:0 10px;font-size:16px;height:28px" title="ダイアログで選択">📂</button>
    </div>
    <div style="flex:1"></div>
    <span class="tr-label" style="font-weight:normal">使用モデル</span>
    <select id="at-model-sel"></select>
    <div class="hdr-sep"></div>
    <span class="tr-label" style="font-weight:normal">閾値</span>
    <input type="range" id="at-thr" min="0.1" max="0.95" step="0.05" value="0.35"
      oninput="document.getElementById('at-thr-disp').textContent=parseFloat(this.value).toFixed(2)">
    <span id="at-thr-disp">0.35</span>
    <button id="at-run" onclick="runAutotag()">タグ付け</button>
    <div id="at-progress-wrap"><div id="at-progress-bar"></div></div>
  </div>
</div>

<div id="main">
  <div id="sim-preview">
    <div id="sim-toolbar">
      <span id="sim-filename">単体画像</span>
    </div>
    <div id="sim-drop" onclick="openFilePicker()">
      <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.2"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/><path d="m21 15-5-5L5 21"/></svg>
      <span>クリックして画像を選択<br>またはここにD&amp;D</span>
    </div>
    <img id="sim-img" onclick="openFilePicker()" title="クリックして別の画像を選択"
      ondragover="event.preventDefault()" ondrop="handleSimDrop(event)" />
  </div>
  <div id="gp">
    <div id="gtb">
      <label><input type="checkbox" id="chk-all" onchange="toggleAll(this.checked)"> 全て</label>
      <span id="sel-count">0 選択</span>
    </div>
    <div id="grid-wrap">
      <div id="grid"></div>
      <div id="drop-ov"><span>📷 画像をドロップして追加</span></div>
    </div>
  </div>

  <div id="rp">
    <div id="empty-st">
      <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.2">
        <rect x="3" y="3" width="18" height="18" rx="2"/><path d="M3 9h18M9 21V9"/>
      </svg>
      <span>画像を選択してください</span>
    </div>

    <!-- 編集パネル (単体・複数共通) -->
    <div id="single-pnl" style="display:none">
      <div id="single-hdr">
        <span class="tr-label">トリガーワード</span>
        <input id="trigger-input" class="tr-input" />
        <button class="tr-btn" onclick="addTrigger('prepend')">先頭</button>
        <button class="tr-btn" onclick="addTrigger('append')">末尾</button>
        <div style="flex:1"></div>
        <button id="txt-toggle" class="te-btn" onclick="toggleTextMode()">テキスト</button>
        <button class="te-btn danger" onclick="clearAllTags()">全削除</button>
        <button class="te-btn ok" onclick="saveSingle()">保存</button>
      </div>
      <div id="chips-outer">
        <div id="chips-area" onclick="document.getElementById('tag-add-input').focus()"></div>
        <div id="tag-input-wrap">
          <span style="font-size:14px;color:var(--txt4);font-weight:600">＋</span>
          <input id="tag-add-input" placeholder="タグを入力して Enter（カンマで複数追加）"
            onkeydown="onTagKey(event)" />
        </div>
      </div>
      <textarea id="tag-textarea" oninput="onTextAreaInput()"></textarea>
    </div>

  </div>
</div>

<div id="toast"></div>

<script>
const A='/tag-editor/api';
let images=[],selected=null,checked=new Set();
let currentTags=[],isDirty=false,modelStatus={};
let pendingTags={}; // path → tags[], auto-tag results not yet saved to disk

// ── モード切り替え ────────────────────────────────────
let mode='dir';
// モードごとの保存状態
const modeState={
  dir:{selected:null,currentTags:[],isDirty:false,checked:new Set(),images:[],pendingTags:{}},
  single:{selected:null,currentTags:[],isDirty:false,pendingTags:{}}
};
function saveState(){
  modeState[mode].selected=selected;
  modeState[mode].currentTags=[...currentTags];
  modeState[mode].isDirty=isDirty;
  modeState[mode].pendingTags={...pendingTags};
  if(mode==='dir'){modeState.dir.checked=new Set(checked);modeState.dir.images=[...images];}
}
function restoreState(m){
  const s=modeState[m];
  selected=s.selected;currentTags=[...s.currentTags];isDirty=s.isDirty;pendingTags={...s.pendingTags};
  if(m==='dir'){checked=new Set(s.checked);images=[...s.images];}
  else{checked.clear();}
}
function setMode(m){
  if(mode===m)return;
  saveState();
  mode=m;
  restoreState(m);
  const isDir=m==='dir';
  // タブ・パネル切り替え
  document.getElementById('tab-dir').classList.toggle('active',isDir);
  document.getElementById('tab-single').classList.toggle('active',!isDir);
  document.getElementById('dir-row').style.display=isDir?'flex':'none';
  document.getElementById('gp').style.display=isDir?'flex':'none';
  document.getElementById('sim-preview').style.display=isDir?'none':'flex';
  if(isDir){
    renderGrid();
    if(checked.size>=2){showPane('single');showMultiChips();}
    else if(selected){showPane('single');renderChips();}
    else{showPane('single');renderChips();}
  }else{
    if(selected){
      const img=document.getElementById('sim-img');
      img.src=`${A}/image?path=${enc(selected)}`;
      img.style.display='block';
      document.getElementById('sim-drop').style.display='none';
      document.getElementById('sim-filename').textContent=selected.split(/[\\/]/).pop();
      showPane('single');renderChips();
    }else{
      document.getElementById('sim-img').style.display='none';
      document.getElementById('sim-drop').style.display='flex';
      document.getElementById('sim-filename').textContent='単体画像';
      showPane('empty');
    }
  }
}

// ── D&D ──────────────────────────────────────────────
function parsePaths(uriList){
  return uriList.split(/\r?\n/).map(u=>u.trim())
    .filter(u=>u.startsWith('file:'))
    .map(u=>decodeURIComponent(u.replace(/^file:\/\/\//,'').replace(/^file:\/\//,'')).replace(/\//g,'\\'));
}
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
  if(paths.length>0){
    if(mode==='single'){simLoad(paths[0]);return;}
    await addPathsToGrid(paths);return;
  }
  if(e.dataTransfer.files.length>0&&mode==='single')openFilePicker();
});

async function addPathsToGrid(paths){
  const r=await fetch(`${A}/validate-paths`,{method:'POST',
    headers:{'Content-Type':'application/json'},body:JSON.stringify({paths})});
  const d=await r.json();
  let added=0;
  (d.valid||[]).forEach(p=>{
    if(images.some(i=>i.path===p))return;
    images.push({path:p,name:p.split(/[\\/]/).pop(),hasTags:false});added++;
  });
  if(added>0){renderGrid();toast(`${added} 枚追加`,'ok');}
  else toast('追加できる画像がありませんでした');
}

// ── ファイルピッカー ──────────────────────────────────
async function openFilePicker(){
  try{
    const r=await fetch(`${A}/pick-files`);
    if(!r.ok)throw new Error(`HTTP ${r.status}`);
    const d=await r.json();
    if(d.error)return toast(`エラー: ${d.error}`,'err');
    if(!d.files||d.files.length===0)return;
    if(mode==='single'){simLoad(d.files[0]);return;}
    await addPathsToGrid(d.files);
  }catch(e){toast(`エラー: ${e.message}`,'err');}
}

function handleSimDrop(e){
  e.preventDefault();e.stopPropagation();
  const paths=parsePaths(e.dataTransfer.getData('text/uri-list')||'');
  if(paths.length>0){simLoad(paths[0]);return;}
  if(e.dataTransfer.files.length>0)openFilePicker();
}

// ── 単体画像 ──────────────────────────────────────────
async function simLoad(path){
  const img=document.getElementById('sim-img');
  const drop=document.getElementById('sim-drop');
  img.src=`${A}/image?path=${enc(path)}`;
  img.style.display='block';drop.style.display='none';
  document.getElementById('sim-filename').textContent=path.split(/[\\/]/).pop();
  selected=path;
  const r=await fetch(`${A}/tags?path=${enc(path)}`);
  const d=await r.json();
  currentTags=parseTags(d.tags||'');isDirty=false;
  renderChips();showPane('single');
}
function resetSim(){
  document.getElementById('sim-img').style.display='none';
  document.getElementById('sim-drop').style.display='flex';
  document.getElementById('sim-filename').textContent='単体画像';
  selected=null;currentTags=[];isDirty=false;
  showPane('empty');
}

// ── フォルダ ──────────────────────────────────────────
async function pickDir(){
  try{
    const r=await fetch(`${A}/pick-dir`);
    if(!r.ok)throw new Error(`HTTP ${r.status}`);
    const d=await r.json();
    if(d.error)return toast(`エラー: ${d.error}`,'err');
    if(d.dir){document.getElementById('dir-input').value=d.dir;loadDir();}
  }catch(e){toast(`エラー: ${e.message}`,'err');}
}
async function loadDir(){
  const dir=document.getElementById('dir-input').value.trim();
  if(!dir)return toast('フォルダを指定してください','err');
  saveConfig({last_dir:dir});
  try{
    const r=await fetch(`${A}/images?dir=${enc(dir)}`);
    if(!r.ok)throw new Error(`HTTP ${r.status}`);
    const d=await r.json();
    if(d.error)return toast(d.error,'err');
    images=d.images;checked.clear();
    selected=null;
    renderGrid();
    updSel();
  }catch(e){toast(`エラー: ${e.message}`,'err');}
}

// ── トリガーワード ────────────────────────────────────
async function addTrigger(pos){
  const words=parseTags(document.getElementById('trigger-input').value);
  if(!words.length)return toast('トリガーワードを入力してください','err');
  const label=pos==='prepend'?'先頭':'末尾';
  function applyWords(tags){
    if(pos==='prepend'){
      const toAdd=words.filter(w=>!tags.includes(w));
      return[...toAdd,...tags];
    }else{
      const base=[...tags];words.forEach(w=>{if(!base.includes(w))base.push(w);});return base;
    }
  }
  if(checked.size>=2){
    await Promise.all([...checked].map(async p=>{
      let tags=pendingTags[p]!==undefined?[...pendingTags[p]]
        :parseTags((await fetch(`${A}/tags?path=${enc(p)}`).then(r=>r.json())).tags||'');
      pendingTags[p]=applyWords(tags);
    }));
    toast(`${words.join(', ')} を${label}に追加`,'ok');
    await showMultiChips();
  }else if(selected){
    currentTags=applyWords(currentTags);
    isDirty=true;renderChips();
    toast(`${words.join(', ')} を${label}に追加`,'ok');
  }else{
    toast('画像を選択してください','err');
  }
}

// ── グリッド ──────────────────────────────────────────
function renderGrid(){
  const g=document.getElementById('grid');g.innerHTML='';
  images.forEach(img=>{
    const w=document.createElement('div');
    w.className='tw'+(img.path===selected?' sel':'')+(checked.has(img.path)?' chk':'');
    w.dataset.path=img.path;
    w.onclick=e=>{if(e.target.type==='checkbox')return;selImg(img.path);};
    const c=document.createElement('input');c.type='checkbox';c.className='tchk';
    c.checked=checked.has(img.path);
    c.onchange=()=>{
      if(c.checked)checked.add(img.path);else checked.delete(img.path);
      w.classList.toggle('chk',c.checked);updSel();
    };
    const im=document.createElement('img');
    im.src=`${A}/image?path=${enc(img.path)}`;im.loading='lazy';im.alt=img.name;
    w.appendChild(c);w.appendChild(im);
    g.appendChild(w);
  });
  updSel();
}

// ── パネル ────────────────────────────────────────────
function showPane(p){
  document.getElementById('empty-st').style.display=p==='empty'?'flex':'none';
  document.getElementById('single-pnl').style.display=p==='single'?'flex':'none';
}

// ── 単体選択・タグ編集 ────────────────────────────────
async function selImg(path){
  selected=path;
  document.querySelectorAll('.tw').forEach(w=>w.classList.toggle('sel',w.dataset.path===path));
  if(pendingTags[path]!==undefined){
    currentTags=[...pendingTags[path]];isDirty=true;
  }else{
    const r=await fetch(`${A}/tags?path=${enc(path)}`);const d=await r.json();
    currentTags=parseTags(d.tags||'');isDirty=false;
  }
  showPane('single');
  if(checked.size>=2)showMultiChips();else renderChips();
}

async function showMultiChips(){
  const area=document.getElementById('chips-area');
  area.innerHTML='<span style="color:var(--txt5);font-size:13px">読み込み中...</span>';
  const total=checked.size;
  const map=new Map();
  await Promise.all([...checked].map(async p=>{
    const tags=pendingTags[p]!==undefined?pendingTags[p]
      :await fetch(`${A}/tags?path=${enc(p)}`).then(r=>r.json()).then(d=>parseTags(d.tags||''));
    tags.forEach(t=>map.set(t,(map.get(t)||0)+1));
  }));
  area.innerHTML='';
  [...map.entries()].sort((a,b)=>b[1]-a[1]).forEach(([tag,cnt])=>{
    const chip=document.createElement('div');chip.className='chip';

    const tagEsc=tag.replace(/\\/g,'\\\\').replace(/'/g,"\\'");
    chip.innerHTML=`<span>${esc(tag)}</span><span class="chip-cnt">${cnt}/${total}</span>`+
      `<span class="chip-x" onclick="multiRmChip('${tagEsc}')">✕</span>`;
    area.appendChild(chip);
  });
}
async function multiRmChip(tag){
  await Promise.all([...checked].map(async p=>{
    let tags=pendingTags[p]!==undefined?[...pendingTags[p]]
      :parseTags((await fetch(`${A}/tags?path=${enc(p)}`).then(r=>r.json())).tags||'');
    pendingTags[p]=tags.filter(t=>t!==tag);
  }));
  if(selected&&checked.has(selected))currentTags=[...pendingTags[selected]];
  await showMultiChips();
}

let textMode=false;
function toggleTextMode(){
  textMode=!textMode;
  const btn=document.getElementById('txt-toggle');
  const outer=document.getElementById('chips-outer');
  const ta=document.getElementById('tag-textarea');
  btn.classList.toggle('active',textMode);
  if(textMode){
    outer.style.display='none';
    ta.style.display='block';
    ta.value=currentTags.join(', ');
    ta.focus();
  }else{
    ta.style.display='none';
    outer.style.display='flex';
    const parsed=parseTags(ta.value);
    if(parsed.join(',')||!currentTags.length){
      currentTags=parsed;isDirty=true;
      if(checked.size>=2&&selected)pendingTags[selected]=[...currentTags];
      else if(selected)pendingTags[selected]=[...currentTags];
    }
    renderChips();
  }
}
function onTextAreaInput(){
  currentTags=parseTags(document.getElementById('tag-textarea').value);
  isDirty=true;
  if(selected)pendingTags[selected]=[...currentTags];
}
function renderChips(){
  if(textMode){document.getElementById('tag-textarea').value=currentTags.join(', ');return;}
  const area=document.getElementById('chips-area');area.innerHTML='';
  currentTags.forEach(tag=>{
    const chip=document.createElement('div');chip.className='chip';
    const tagEsc=tag.replace(/\\/g,'\\\\').replace(/'/g,"\\'");
    chip.innerHTML=`<span>${esc(tag)}</span><span class="chip-x" onclick="removeChip('${tagEsc}')">✕</span>`;
    area.appendChild(chip);
  });
}

async function onTagKey(e){
  const inp=e.target;
  if(e.key==='Enter'){
    e.preventDefault();
    const raw=inp.value.trim();if(!raw)return;
    if(checked.size>=2){
      const newTags=parseTags(raw);
      await Promise.all([...checked].map(async p=>{
        let tags=pendingTags[p]!==undefined?[...pendingTags[p]]
          :parseTags((await fetch(`${A}/tags?path=${enc(p)}`).then(r=>r.json())).tags||'');
        newTags.forEach(t=>{if(!tags.includes(t))tags.push(t);});
        pendingTags[p]=tags;
      }));
      if(selected&&checked.has(selected))currentTags=[...pendingTags[selected]];
      await showMultiChips();
    }else{
      parseTags(raw).forEach(t=>{if(t&&!currentTags.includes(t))currentTags.push(t);});
      isDirty=true;renderChips();
    }
    inp.value='';
  }else if(e.key==='Backspace'&&inp.value===''&&currentTags.length>0&&checked.size<2){
    currentTags.pop();isDirty=true;renderChips();
  }
}

function removeChip(tag){
  currentTags=currentTags.filter(t=>t!==tag);isDirty=true;
  renderChips();
}

function _updateThumbDot(path,tags){
  const img=images.find(i=>i.path===path);
  if(!img)return;
  img.hasTags=tags.length>0;delete pendingTags[path];
}



async function saveSingle(silent=false){
  if(checked.size>=2){
    // 複数選択: pendingTagsがある画像のみ保存（変更のあるもの）
    const toSave=[...checked].filter(p=>pendingTags[p]!==undefined);
    if(!toSave.length){if(!silent)toast('保存する変更がありません');return;}
    await Promise.all(toSave.map(async p=>{
      const tags=pendingTags[p];
      await fetch(`${A}/save`,{method:'POST',headers:{'Content-Type':'application/json'},
        body:JSON.stringify({path:p,tags:tags.join(', ')})});
      _updateThumbDot(p,tags);
    }));
    isDirty=false;renderGrid();
    if(!silent)toast(`${toSave.length}枚保存しました`,'ok');
  }else{
    if(!selected)return;
    await fetch(`${A}/save`,{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({path:selected,tags:currentTags.join(', ')})});
    _updateThumbDot(selected,currentTags);
    isDirty=false;if(!silent)toast('保存しました','ok');
  }
}

async function clearAllTags(){
  if(!confirm('このタグを全て削除しますか？'))return;
  if(checked.size>=2){
    await Promise.all([...checked].map(p=>{pendingTags[p]=[];}));
    await showMultiChips();
  }else{
    currentTags=[];isDirty=true;
    if(selected)pendingTags[selected]=[];
    renderChips();
  }
}



// ── 選択管理 ──────────────────────────────────────────
function selAll(){
  images.forEach(i=>checked.add(i.path));
  document.querySelectorAll('.tchk').forEach(c=>c.checked=true);
  document.querySelectorAll('.tw').forEach(w=>w.classList.add('chk'));updSel();
}
function clrSel(){
  checked.clear();
  document.querySelectorAll('.tchk').forEach(c=>c.checked=false);
  document.querySelectorAll('.tw').forEach(w=>w.classList.remove('chk'));updSel();
}
function toggleAll(v){if(v)selAll();else clrSel();}
function updSel(){
  document.getElementById('sel-count').textContent=`${checked.size} 選択`;
  if(checked.size>=2){showPane('single');showMultiChips();}
  else if(selected){showPane('single');renderChips();}
  else if(mode==='dir'){showPane('single');renderChips();}
  else showPane('empty');
}

// ── モデル ────────────────────────────────────────────
let selectedModel='';
async function loadModels(){const r=await fetch(`${A}/models`);modelStatus=await r.json();}
function renderModelSel(){
  const sel=document.getElementById('at-model-sel');
  const prev=sel.value||selectedModel;
  sel.innerHTML='<option value="">-- 選択 --</option>';
  Object.entries(modelStatus).forEach(([name,info])=>{
    const opt=document.createElement('option');
    opt.value=name;
    opt.textContent=info.label+(info.ready?'':' (未DL)');
    opt.title=info.desc;
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
      if(confirm(`「${info.label}」はダウンロードが必要です。\nダウンロードしますか？`)){
        await startDownload(name);
      }else{sel.value=selectedModel;return;}
    }
    selectedModel=name;
    saveConfig({selected_models:[name]});
  };
}
function getCheckedModels(){return selectedModel?[selectedModel]:[];}
async function startDownload(name){
  const info=modelStatus[name];
  toast(`${info.label} DL中...`);
  await fetch(`${A}/download`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({model:name})});
  while(true){
    await new Promise(r=>setTimeout(r,1500));
    const r=await fetch(`${A}/dl-status?model=${name}`);const s=await r.json();
    if(s.state==='done'){
      modelStatus[name].ready=true;
      selectedModel=name;saveConfig({selected_models:[name]});
      renderModelSel();
      document.getElementById('at-model-sel').value=name;
      toast(`${info.label} DL完了`,'ok');break;
    }else if(s.state==='error'){
      toast(`DL失敗: ${s.error}`,'err');break;
    }
  }
}

// ── 自動タグ付け ──────────────────────────────────────
async function runAutotag(){
  const models=getCheckedModels();
  if(!models.length)return toast('モデルを選択してください','err');
  const targets=mode==='single'?(selected?[selected]:[]):(checked.size>0?[...checked]:[]);
  if(!targets.length)return toast('画像を選択してください','err');
  const notReady=models.filter(m=>!modelStatus[m]?.ready);
  if(notReady.length)return toast(`未DLのモデル: ${notReady.join(', ')}`,'err');
  const btn=document.getElementById('at-run');
  const wrap=document.getElementById('at-progress-wrap');
  const bar=document.getElementById('at-progress-bar');
  btn.disabled=true;wrap.style.display='block';bar.style.width='0%';
  const thr=parseFloat(document.getElementById('at-thr').value);
  let done=0;
  for(const path of targets){
    bar.style.width=`${Math.round(done/targets.length*100)}%`;
    try{
      const r=await fetch(`${A}/autotag`,{method:'POST',headers:{'Content-Type':'application/json'},
        body:JSON.stringify({paths:[path],models,threshold:thr,merge_mode:'replace'})});
      const d=await r.json();
      if(d.ok&&d.results.length>0)pendingTags[path]=d.results[0].tags;
    }catch(e){}
    done++;
  }
  bar.style.width='100%';
  setTimeout(()=>{
    wrap.style.display='none';bar.style.width='0%';btn.disabled=false;
  },1200);
  renderGrid();
  if(selected&&pendingTags[selected]!==undefined){
    currentTags=[...pendingTags[selected]];isDirty=true;renderChips();
  }
  if(checked.size>=2)showMultiChips();
}

// ── ユーティリティ ────────────────────────────────────
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

(async()=>{
  try{
    await loadModels();
    const r=await fetch(`${A}/config`);
    if(!r.ok)throw new Error(`API ${r.status}`);
    const cfg=await r.json();
    if(cfg.selected_models&&cfg.selected_models[0]&&modelStatus[cfg.selected_models[0]])
      selectedModel=cfg.selected_models[0];
    renderModelSel();
    if(cfg.last_dir){document.getElementById('dir-input').value=cfg.last_dir;await loadDir();}
    else{showPane('single');renderChips();}
  }catch(e){
    toast(`API接続エラー: ${e.message}`,'err');
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
            folder = filedialog.askdirectory(title="画像フォルダを選択")
            root.destroy()
            return JSONResponse({"dir": folder or ""})
        except Exception as e:
            return JSONResponse({"dir": "", "error": str(e)})

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
                title="画像ファイルを選択",
                filetypes=[("画像ファイル", "*.png *.jpg *.jpeg *.webp *.gif *.bmp"), ("すべて", "*.*")]
            )
            root.destroy()
            return JSONResponse({"files": list(files)})
        except Exception as e:
            return JSONResponse({"files": [], "error": str(e)})

    @app.get("/tag-editor/api/images")
    async def list_images(dir: str = ""):
        p = Path(dir)
        if not p.exists() or not p.is_dir():
            return JSONResponse({"error": "フォルダが見つかりません"})
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
        if not p.exists(): return JSONResponse({"error": "ファイルが見つかりません"})
        return JSONResponse({"tags": _read_tags(p)})

    @app.post("/tag-editor/api/save")
    async def save_tags(req: Request):
        data = await req.json()
        p = Path(data.get("path", ""))
        if not p.exists(): return JSONResponse({"error": "ファイルが見つかりません"})
        _write_tags(p, data.get("tags", ""))
        return JSONResponse({"ok": True})

    @app.get("/tag-editor/api/models")
    async def get_models():
        result = {}
        for name, info in TAGGER_MODELS.items():
            result[name] = {
                "repo": info["repo"], "label": info["label"],
                "desc": info["desc"], "ready": _model_ready(name),
            }
        return JSONResponse(result)

    @app.post("/tag-editor/api/download")
    async def start_download(req: Request):
        data = await req.json()
        name = data.get("model", "")
        if name not in TAGGER_MODELS:
            return JSONResponse({"error": "不明なモデル"})
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
                return JSONResponse({"error": f"モデル未DL: {m}"})

        try:
            results = []
            for path_str in paths:
                p = Path(path_str)
                if not p.exists(): continue

                seen = []
                seen_set = set()
                for m in models:
                    for tag in _run_tagger(m, p, threshold):
                        if tag not in seen_set:
                            seen.append(tag)
                            seen_set.add(tag)

                results.append({"path": path_str, "tags": seen})

            return JSONResponse({"ok": True, "count": len(results), "results": results})
        except Exception as e:
            return JSONResponse({"error": str(e)})

    @app.post("/tag-editor/api/batch")
    async def batch_op(req: Request):
        data = await req.json()
        op = data.get("op")
        paths = data.get("paths", [])
        count = 0
        updated = []
        for path_str in paths:
            p = Path(path_str)
            if not p.exists(): continue
            tags = _parse_tags(_read_tags(p))
            if op == "add":
                for t in data.get("tags", []):
                    if t not in tags: tags.append(t)
            elif op == "prepend":
                new = [t for t in data.get("tags", []) if t not in tags]
                tags = new + tags
            elif op == "remove":
                rm = set(data.get("tags", []))
                tags = [t for t in tags if t not in rm]
            elif op == "replace":
                find, rep = data.get("find", ""), data.get("replace", "")
                tags = [rep if t == find else t for t in tags]
                tags = [t for t in tags if t]
            elif op == "sort":
                tags = sorted(tags)
            elif op == "dedup":
                seen = []; [seen.append(t) for t in tags if t not in seen]; tags = seen
            _write_tags(p, ", ".join(tags))
            updated.append({"path": path_str, "hasTags": len(tags) > 0})
            count += 1
        return JSONResponse({"ok": True, "count": count, "updated": updated})


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
