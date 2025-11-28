#setup_fixed2.sh

#!/bin/bash
set -e # エラーが発生した場合、スクリプトの実行を即座に停止します

# --- このスクリプトの目的 ---
# Google Colab環境で、condaを使わずにpipとaptのみで
# feature_splattingプロジェクトに必要なすべての依存関係をインストールする
# -----------------------------

echo "=== Colab環境セットアップ開始 ==="

# --- 1. システムパッケージの更新とインストール ---
echo "--- 1. システムパッケージの更新 ---"
apt-get update

echo "--- 2. COLMAPとOpenGLライブラリのインストール ---"
apt-get install -y colmap \
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    mesa-utils \
    libegl1-mesa \
    libgbm1 \
    xvfb

# --- 2. Pipパッケージのアップグレード ---
echo "--- 3. Pipツールのアップグレード ---"
pip install --upgrade pip setuptools wheel

# --- 3. PyTorchのインストール ---
echo "--- 4. PyTorchのインストール ---"
# Colabのデフォルトは通常CUDA 11.xまたは12.x対応版
# 既存のtorchが入っている場合は上書きされる
pip install torch torchvision torchaudio

# --- 4. ビルドツールのインストール ---
echo "--- 5. Ninjaビルドシステムのインストール ---"
pip install ninja

# --- 5. 基本的なPythonパッケージのインストール ---
echo "--- 6. 基本的なPythonパッケージのインストール ---"
pip install opencv-python natsort scipy kornia plyfile Pillow tqdm


# --- 7. Gaussian Splatting カスタム拡張機能のビルド ---
echo "--- 8. Gaussian Splatting カスタム拡張機能のビルド ---"
echo "    (この処理には数分かかる場合があります)"

pip install thirdparty/gaussian_splatting/submodules/gaussian_rasterization_ch3
pip install thirdparty/gaussian_splatting/submodules/gaussian_rasterization_ch9
pip install thirdparty/gaussian_splatting/submodules/forward_lite
pip install thirdparty/gaussian_splatting/submodules/forward_full

# --- 8. MMCVのインストール ---
echo "--- 9. MMCVのインストール (時間がかかります) ---"
pip install -e thirdparty/mmcv -v


# --- 6. Simple-KNNのインストール ---
echo "--- 7. Simple-KNNのインストール ---"
pip install git+https://github.com/camenduru/simple-knn.git


# --- 9. NumPy互換性修正 ---
echo "--- 10. NumPy互換性の確認と修正 ---"
python -c "import numpy as np; print(f'NumPy version: {np.__version__}')"

# np.NaN を np.nan に置換
if [ -f "/content/SpacetimeGaussians/thirdparty/colmap/pre_colmap.py" ]; then
    echo "    pre_colmap.py の修正..."
    sed -i 's/np\.NaN/np.nan/g' /content/SpacetimeGaussians/thirdparty/colmap/pre_colmap.py
    grep "np\.nan" /content/SpacetimeGaussians/thirdparty/colmap/pre_colmap.py | head -n 5 || echo "    (該当行なし)"
fi

if [ -f "/content/SpacetimeGaussians/thirdparty/colmap/pre_colmap_fixed1.py" ]; then
    echo "    pre_colmap_fixed1.py の修正..."
    sed -i 's/np\.NaN/np.nan/g' /content/SpacetimeGaussians/thirdparty/colmap/pre_colmap_fixed1.py
    grep "np\.nan" /content/SpacetimeGaussians/thirdparty/colmap/pre_colmap_fixed1.py | head -n 5 || echo "    (該当行なし)"
fi

# --- 10. 環境変数の設定 ---
echo "--- 11. OpenGL/Qt環境変数の設定 ---"
export QT_QPA_PLATFORM='offscreen'
export QT_QPA_PLATFORM_PLUGIN_PATH=''
export QT_PLUGIN_PATH=''
export LIBGL_ALWAYS_SOFTWARE='1'
export QT_DEBUG_PLUGINS='0'

# 環境変数を永続化 (現在のセッション用)
echo "export QT_QPA_PLATFORM='offscreen'" >> ~/.bashrc
echo "export QT_QPA_PLATFORM_PLUGIN_PATH=''" >> ~/.bashrc
echo "export QT_PLUGIN_PATH=''" >> ~/.bashrc
echo "export LIBGL_ALWAYS_SOFTWARE='1'" >> ~/.bashrc
echo "export QT_DEBUG_PLUGINS='0'" >> ~/.bashrc

echo ""
echo "=== セットアップ完了 ==="
echo "インストールされたパッケージの確認:"
echo "- PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "- CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "- OpenCV: $(python -c 'import cv2; print(cv2.__version__)')"
echo ""
echo "環境変数が設定されました。Pythonスクリプトでは以下を実行してください:"
echo ""
echo "import os"
echo "os.environ['QT_QPA_PLATFORM'] = 'offscreen'"
echo "os.environ['LIBGL_ALWAYS_SOFTWARE'] = '1'"
echo ""
echo "準備完了です!"
