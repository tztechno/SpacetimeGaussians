#setup_fixed4.sh

#!/bin/bash
set -e # エラーが発生した場合、スクリプトの実行を即座に停止します

# --- このスクリプトの目的 ---
# Google Colab環境で、condaを使わずにpipとaptのみで
# feature_splattingプロジェクトに必要なすべての依存関係をインストールする
# -----------------------------

echo "=== Colab環境セットアップ開始 ==="

# プロジェクトルートを設定
PROJECT_ROOT="$(pwd)"
echo "プロジェクトルート: $PROJECT_ROOT"

# --- 1. システムパッケージの更新とビルドツールのインストール ---
echo "--- 1. システムパッケージの更新とビルドツールのインストール ---"
apt-get update
apt-get install -y build-essential ninja-build

# --- 2. COLMAPとOpenGLライブラリのインストール ---
echo "--- 2. COLMAPとOpenGLライブラリのインストール ---"
apt-get install -y colmap \
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    mesa-utils \
    libegl1-mesa \
    libgbm1 \
    xvfb

# --- 3. Pipツールのアップグレード ---
echo "--- 3. Pipツールのアップグレード ---"
pip install --upgrade pip setuptools wheel

# --- 4. PyTorchのインストール (CUDA対応版) ---
echo "--- 4. PyTorchのインストール (CUDA 12.1対応版) ---"
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# PyTorchとCUDAの確認
echo "--- PyTorchとCUDAの確認 ---"
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"

# --- 5. ビルド用Pythonパッケージのインストール ---
echo "--- 5. ビルド用Pythonパッケージのインストール ---"
pip install ninja packaging wheel
sudo apt-get update
sudo apt-get install -y build-essential ninja-build

# --- 6. 基本的なPythonパッケージのインストール ---
echo "--- 6. 基本的なPythonパッケージのインストール ---"
pip install opencv-python natsort scipy kornia plyfile Pillow tqdm

# --- 7. CUDA環境変数の設定 ---
echo "--- 7. CUDA環境変数の設定 ---"
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# --- 8. Simple-KNNのインストール ---
echo "--- 8. Simple-KNNのインストール ---"

# プロジェクトルートを保存
PROJECT_ROOT="$(pwd)"
SCRIPT_DIR="$PROJECT_ROOT/script"

# simple-knn-main.zip の場所を確認 (scriptフォルダまたはthirdpartyフォルダ)
if [ -f "$SCRIPT_DIR/simple-knn-main.zip" ]; then
    SIMPLE_KNN_ZIP="$SCRIPT_DIR/simple-knn-main.zip"
    echo "    simple-knn-main.zip を script フォルダから読み込みます"
elif [ -f "$PROJECT_ROOT/thirdparty/simple-knn-main.zip" ]; then
    SIMPLE_KNN_ZIP="$PROJECT_ROOT/thirdparty/simple-knn-main.zip"
    echo "    simple-knn-main.zip を thirdparty フォルダから読み込みます"
else
    echo "ERROR: simple-knn-main.zip が見つかりません"
    echo "       script/ または thirdparty/ フォルダに配置してください"
    exit 1
fi

TMP_DIR="$PROJECT_ROOT/simple-knn-tmp"

rm -rf "$TMP_DIR"
mkdir -p "$TMP_DIR"

unzip -q "$SIMPLE_KNN_ZIP" -d "$TMP_DIR"

cd "$TMP_DIR/simple-knn-main"
pip install --no-build-isolation .

# プロジェクトルートに戻る
cd "$PROJECT_ROOT"
rm -rf "$TMP_DIR"

echo "--- Simple-KNN installation completed ---"

# --- 9. MMCVのインストール ---
echo "--- 9. MMCVのインストール (時間がかかります) ---"

# プロジェクトルートにいることを確認
cd "$PROJECT_ROOT"

# thirdparty/mmcv が存在するか確認
if [ -d "thirdparty/mmcv" ]; then
    echo "    thirdparty/mmcv が見つかりました。インストールを開始します..."
    pip install -e thirdparty/mmcv -v
else
    echo "    警告: thirdparty/mmcv が見つかりません。スキップします..."
    echo "    もし必要な場合は、mmcvを手動でインストールしてください:"
    echo "    pip install mmcv==2.0.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html"
fi

# --- 10. Gaussian Splatting カスタム拡張機能のビルド ---
echo "--- 10. Gaussian Splatting カスタム拡張機能のビルド ---"
echo "    (この処理には数分かかる場合があります)"

# 各サブモジュールが存在するか確認してからビルド
if [ -d "thirdparty/gaussian_splatting/submodules/gaussian_rasterization_ch3" ]; then
    echo "    - gaussian_rasterization_ch3 をビルド中..."
    pip install -v -e thirdparty/gaussian_splatting/submodules/gaussian_rasterization_ch3 --no-build-isolation
else
    echo "    警告: gaussian_rasterization_ch3 が見つかりません"
fi

if [ -d "thirdparty/gaussian_splatting/submodules/gaussian_rasterization_ch9" ]; then
    echo "    - gaussian_rasterization_ch9 をビルド中..."
    pip install -v -e thirdparty/gaussian_splatting/submodules/gaussian_rasterization_ch9 --no-build-isolation
else
    echo "    警告: gaussian_rasterization_ch9 が見つかりません"
fi

if [ -d "thirdparty/gaussian_splatting/submodules/forward_lite" ]; then
    echo "    - forward_lite をビルド中..."
    pip install -v -e thirdparty/gaussian_splatting/submodules/forward_lite --no-build-isolation
else
    echo "    警告: forward_lite が見つかりません"
fi

if [ -d "thirdparty/gaussian_splatting/submodules/forward_full" ]; then
    echo "    - forward_full をビルド中..."
    pip install -v -e thirdparty/gaussian_splatting/submodules/forward_full --no-build-isolation
else
    echo "    警告: forward_full が見つかりません"
fi

# --- 11. NumPy互換性修正 ---
echo "--- 11. NumPy互換性の確認と修正 ---"
python -c "import numpy as np; print(f'NumPy version: {np.__version__}')"

# np.NaN を np.nan に置換
if [ -f "thirdparty/colmap/pre_colmap.py" ]; then
    echo "    pre_colmap.py の修正..."
    sed -i 's/np\.NaN/np.nan/g' thirdparty/colmap/pre_colmap.py
    grep "np\.nan" thirdparty/colmap/pre_colmap.py | head -n 5 || echo "    (該当行なし)"
fi

if [ -f "thirdparty/colmap/pre_colmap_fixed1.py" ]; then
    echo "    pre_colmap_fixed1.py の修正..."
    sed -i 's/np\.NaN/np.nan/g' thirdparty/colmap/pre_colmap_fixed1.py
    grep "np\.nan" thirdparty/colmap/pre_colmap_fixed1.py | head -n 5 || echo "    (該当行なし)"
fi

# --- 12. OpenGL/Qt環境変数の設定 ---
echo "--- 12. OpenGL/Qt環境変数の設定 ---"
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
echo "export CUDA_HOME=/usr/local/cuda" >> ~/.bashrc
echo "export PATH=\$CUDA_HOME/bin:\$PATH" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH" >> ~/.bashrc

echo ""
echo "=== セットアップ完了 ==="
echo "インストールされたパッケージの確認:"
echo "- PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "- CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "- CUDA version: $(python -c 'import torch; print(torch.version.cuda)')"
echo "- OpenCV: $(python -c 'import cv2; print(cv2.__version__)')"
echo ""
echo "環境変数が設定されました。Pythonスクリプトでは以下を実行してください:"
echo ""
echo "import os"
echo "os.environ['QT_QPA_PLATFORM'] = 'offscreen'"
echo "os.environ['LIBGL_ALWAYS_SOFTWARE'] = '1'"
echo ""
echo "準備完了です!"
