# sd-webui-taggitor

Stable Diffusion WebUI Forge NEO 向けのタグ管理拡張機能です

![screenshot](screenshot.png)

## 機能

### 参照モード
- 画像をD&Dまたはファイル選択で読み込み、タグを確認
- WD14モデルで解析し、結果を表示

### 編集モード
- フォルダ内の画像を一覧表示
- チップ形式のタグ編集UI（追加・削除）
- テキスト形式でのタグ直接編集
- トリガーワードを先頭または末尾に一括追加
- 複数画像選択時の共通タグ表示・一括削除・一括保存
- WD14モデルで解析してタグを反映

## インストール方法

1. WebUI を起動
2. **Extensions** タブ → **Install from URL** を開く
3. 以下のURLを貼り付けて Install をクリック：
   ```
   https://github.com/ranran141/sd-webui-taggitor
   ```
4. WebUIを再起動
5. **Taggitor** タブが作成されていればインストール完了

## 動作環境

- Stable Diffusion WebUI Forge NEO

## 解析モデルについて

[SmilingWolf](https://huggingface.co/SmilingWolf) 氏が公開している WD14 ONNX モデルを使用しています。  
初回使用時はUI上のモデル選択から自動でダウンロードできます。

| モデル | 特徴 |
|--------|------|
| [WD ViT v3](https://huggingface.co/SmilingWolf/wd-vit-tagger-v3) | 汎用・バランス型。速度と精度のバランスが良く、最初の1枚に最適（推奨） |
| [WD SwinV2 v3](https://huggingface.co/SmilingWolf/wd-swinv2-tagger-v3) | 高精度。v3世代の中で安定した検出率 |
| [WD EVA02 Large v3](https://huggingface.co/SmilingWolf/wd-eva02-large-tagger-v3) | 最高精度・低速。精度重視の最終確認向け |
| [WD MOAT v2](https://huggingface.co/SmilingWolf/wd-v1-4-moat-tagger-v2) | 軽量・高速。旧世代だが動作が安定 |

## 更新履歴

### v1.1.0 (2026-05-18)
- 参照モード追加（D&Dで画像を読み込み、タグを確認・解析）
- タグのテキスト直接編集に対応
- UI改善

### v1.0.0 (2026-05-18)
- 初回リリース
