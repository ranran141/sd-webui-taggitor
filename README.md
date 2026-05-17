# sd-webui-taggitor

Stable Diffusion WebUI / Forge Neo / Reforge 向けのタグ編集拡張機能です

![screenshot](screenshot.png)

## 機能

- フォルダ内の画像を一覧表示し、タグをまとめて編集
- 単体画像モードでファイルを直接開いて編集
- WD14 ONNXモデルによる自動タグ付け（複数モデル対応）
- チップ形式のタグ編集UI（追加・削除・一括操作）
- トリガーワードを先頭または末尾に一括追加
- 複数画像選択時の共通タグ表示・一括削除・一括保存

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
- Stable Diffusion WebUI Reforge

## 自動タグ付けのモデルについて

[SmilingWolf](https://huggingface.co/SmilingWolf) 氏が公開している WD14 ONNX モデルを使用しています。  
初回使用時はUI上のモデル選択から自動でダウンロードできます。

対応モデル（HuggingFace より自動取得）：
- [WD ViT v3](https://huggingface.co/SmilingWolf/wd-vit-tagger-v3)（バランス型・推奨）
- [WD SwinV2 v3](https://huggingface.co/SmilingWolf/wd-swinv2-tagger-v3)
- [WD EVA02 Large v3](https://huggingface.co/SmilingWolf/wd-eva02-large-tagger-v3)
- [WD MOAT v2](https://huggingface.co/SmilingWolf/wd-v1-4-moat-tagger-v2)

## 更新履歴

### v1.0.0 (2026-05-18)
- 初回リリース
