#!/bin/bash
set -e # エラーが発生した場合、スクリプトの実行を即座に停止します

# --- このスクリプトの目的 ---
# 1. Conda環境の初期化問題を解決する。
# 2. カスタムCUDA拡張機能のビルド失敗（subprocess-exited-with-error）の原因である、
#    PyTorch/CUDAとGCCコンパイラのバージョン不一致を修正する。
# -----------------------------

# --- 0. 環境の初期化とクリーンアップ ---
echo "--- 0. 既存環境のクリーンアップとConda初期化 ---"
CONDA_ENV="feature_splatting"
COLMAP_ENV="colmapenv"

# 既存の環境があれば削除してクリーンな状態から開始
# 削除エラーは無視 (|| true)
conda env remove -n $CONDA_ENV -y || true
conda env remove -n $COLMAP_ENV -y || true

# Conda初期化のための処理 (Colab環境での一般的な修正)
CONDA_PROFILE="/opt/conda/etc/profile.d/conda.sh"
if [ -f "$CONDA_PROFILE" ]; then
    echo "Condaプロファイルをソースします: $CONDA_PROFILE"
    source "$CONDA_PROFILE"
fi
# シェルフックを評価し、conda activateを有効にする
eval "$(conda shell.bash hook)"


# --- 1. feature_splatting 環境のセットアップ ---
PYTHON_VERSION="3.10"
# カスタム拡張機能のビルドに実績のある安定構成: PyTorch 1.13.1 + CUDA 11.7
PYTORCH_VERSION="1.13.1"
CUDA_VERSION="11.7"
TORCHVISION_VERSION="0.14.1" 

echo "--- 1. $CONDA_ENV 環境の作成 (Python $PYTHON_VERSION) ---"
conda create -n $CONDA_ENV python=$PYTHON_VERSION -y

# feature_splatting をアクティベート
echo "--- feature_splatting 環境をアクティベート ---"
conda activate $CONDA_ENV

echo "--- 2. PyTorch, TorchVision, Torchaudioおよびビルドツールのインストール ---"
# Pytorch 1.13.1 + CUDA 11.7 (互換性を最優先)
conda install pytorch==$PYTORCH_VERSION torchvision==$TORCHVISION_VERSION torchaudio==$PYTORCH_VERSION cudatoolkit=$CUDA_VERSION -c pytorch -c conda-forge -y

# PyTorchカスタム拡張機能のビルドに必須
conda install ninja -c conda-forge -y

# **重要修正:** 特定のGCCバージョンを固定 (ビルド失敗の最も一般的な原因)
# CUDA 11.x系と互換性の高いGCC 9.x系をインストール
echo "--- 2.1. GCC/G++ コンパイラを9.3.0に固定して互換性を確保 ---"
conda install gxx_linux-64=9.3.0 -c conda-forge -y

# channel_priority を設定して、後のpipインストールとの競合を減らす
conda config --set channel_priority false

# --- 3. Pipを使用したカスタム/その他のパッケージのインストール ---
echo "--- 3. カスタム/その他のパッケージのインストール ---"

# ビルドの準備
pip install --upgrade pip setuptools wheel

# --- Gaussian Splatting Custom Extensions ---
echo "--- 3.1. Gaussian Splatting カスタム拡張機能のビルド ---"

# このGCC固定により、ビルドが成功することを期待します。
pip install thirdparty/gaussian_splatting/submodules/gaussian_rasterization_ch9
pip install thirdparty/gaussian_splatting/submodules/gaussian_rasterization_ch3
pip install thirdparty/gaussian_splatting/submodules/forward_full
pip install thirdparty/gaussian_splatting/submodules/forward_lite

# install simpleknn
pip install thirdparty/gaussian_splatting/submodules/simple-knn

# install opencv-python
pip install opencv-python

# Install MMCV for CUDA KNN, used for init point sampling
echo "--- 4. MMCVのインストール (時間がかかります) ---"
pip install -e thirdparty/mmcv -v

# other packages
pip install natsort scipy kornia

echo "--- 5. $CONDA_ENV 環境のパッケージインストール完了 ---"

# --- 6. colmapenv 環境のセットアップ ---
COLMAP_ENV="colmapenv"
COLMAP_PYTHON_VERSION="3.8"

echo "--- 6. $COLMAP_ENV 環境の作成 (Python $COLMAP_PYTHON_VERSION) ---"
conda create -n $COLMAP_ENV python=$COLMAP_PYTHON_VERSION -y

# colmapenv をアクティベート
echo "--- colmapenv 環境をアクティベート ---"
conda activate $COLMAP_ENV

echo "--- 7. Colmap および関連パッケージのインストール ---"
pip install opencv-python-headless tqdm natsort Pillow

# Colmapの依存関係を解決
conda install pytorch==1.12.1 -c pytorch -c conda-forge -y

# Colmap本体のインストール
conda install colmap -c conda-forge -y

echo "--- 8. colmapenv 環境のセットアップ完了 ---"

# --- 9. 最終的に feature_splatting 環境に戻る ---
echo "--- 最終環境: feature_splatting に戻る ---"
conda activate $CONDA_ENV

echo "--- セットアップスクリプトの実行完了 ---"
echo "現在の環境: $CONDA_ENV"