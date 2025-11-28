#setup_fixed5.sh
#!/bin/bash
set -e

echo "=== Kaggle環境セットアップ開始 ==="

# プロジェクトルートを設定
PROJECT_ROOT="$(pwd)"
echo "プロジェクトルート: $PROJECT_ROOT"

# --- 1. システムパッケージの更新 ---
echo "--- 1. システムパッケージの更新 ---"
apt-get update

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

# --- 4. PyTorchの確認 (Kaggleには既にインストール済み) ---
echo "--- 4. PyTorchの確認 ---"
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"

# --- 5. ビルド用パッケージのインストール ---
echo "--- 5. ビルド用パッケージのインストール ---"
pip install ninja packaging wheel

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

SCRIPT_DIR="$PROJECT_ROOT/script"

# simple-knn-main.zip の場所を確認
if [ -f "$SCRIPT_DIR/simple-knn-main.zip" ]; then
    SIMPLE_KNN_ZIP="$SCRIPT_DIR/simple-knn-main.zip"
    echo "    simple-knn-main.zip を script フォルダから読み込みます"
elif [ -f "$PROJECT_ROOT/thirdparty/simple-knn-main.zip" ]; then
    SIMPLE_KNN_ZIP="$PROJECT_ROOT/thirdparty/simple-knn-main.zip"
    echo "    simple-knn-main.zip を thirdparty フォルダから読み込みます"
else
    echo "警告: simple-knn-main.zip が見つかりません。GitHubからダウンロードします..."
    pip install git+https://github.com/camenduru/simple-knn.git
fi

if [ ! -z "$SIMPLE_KNN_ZIP" ]; then
    TMP_DIR="$PROJECT_ROOT/simple-knn-tmp"
    rm -rf "$TMP_DIR"
    mkdir -p "$TMP_DIR"
    unzip -q "$SIMPLE_KNN_ZIP" -d "$TMP_DIR"
    cd "$TMP_DIR/simple-knn-main"
    pip install --no-build-isolation .
    cd "$PROJECT_ROOT"
    rm -rf "$TMP_DIR"
    echo "--- Simple-KNN installation completed ---"
fi

# --- 9. MMCVのビルドとインストール ---
echo "--- 9. MMCVのビルドとインストール (時間がかかります) ---"

cd "$PROJECT_ROOT"

if [ -d "thirdparty/mmcv" ]; then
    echo "    thirdparty/mmcv が見つかりました"
    
    # MMCVのビルドに必要な環境変数を設定
    export MMCV_WITH_OPS=1
    export FORCE_CUDA=1
    
    # setup.pyが存在するか確認
    if [ -f "thirdparty/mmcv/setup.py" ]; then
        echo "    MMCVをビルドしてインストールします..."
        cd thirdparty/mmcv
        
        # クリーンビルド
        rm -rf build dist *.egg-info
        
        # ビルドとインストール
        pip install -e . -v --no-build-isolation
        
        cd "$PROJECT_ROOT"
        echo "    MMCV installation completed"
    else
        echo "    エラー: thirdparty/mmcv/setup.py が見つかりません"
        exit 1
    fi
else
    echo "    エラー: thirdparty/mmcv ディレクトリが見つかりません"
    echo "    プロジェクトのディレクトリ構造を確認してください"
    exit 1
fi

# --- 10. MMCV拡張モジュールの確認 ---
echo "--- 10. MMCV拡張モジュールの確認 ---"
python -c "
try:
    from mmcv import ops
    print('✓ MMCV ops モジュールのインポート成功')
except Exception as e:
    print(f'✗ MMCV ops モジュールのインポート失敗: {e}')
    exit(1)
"

# --- 11. Gaussian Splatting カスタム拡張機能のビルド ---
echo "--- 11. Gaussian Splatting カスタム拡張機能のビルド ---"
echo "    (この処理には数分かかる場合があります)"

cd "$PROJECT_ROOT"

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

# --- 12. NumPy互換性修正 ---
echo "--- 12. NumPy互換性の確認と修正 ---"
python -c "import numpy as np; print(f'NumPy version: {np.__version__}')"

if [ -f "thirdparty/colmap/pre_colmap.py" ]; then
    echo "    pre_colmap.py の修正..."
    sed -i 's/np\.NaN/np.nan/g' thirdparty/colmap/pre_colmap.py
fi

if [ -f "thirdparty/colmap/pre_colmap_fixed1.py" ]; then
    echo "    pre_colmap_fixed1.py の修正..."
    sed -i 's/np\.NaN/np.nan/g' thirdparty/colmap/pre_colmap_fixed1.py
fi

# --- 13. OpenGL/Qt環境変数の設定 ---
echo "--- 13. OpenGL/Qt環境変数の設定 ---"
export QT_QPA_PLATFORM='offscreen'
export QT_QPA_PLATFORM_PLUGIN_PATH=''
export QT_PLUGIN_PATH=''
export LIBGL_ALWAYS_SOFTWARE='1'
export QT_DEBUG_PLUGINS='0'

echo ""
echo "=== セットアップ完了 ==="
echo "インストールされたパッケージの確認:"
echo "- PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "- CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "- CUDA version: $(python -c 'import torch; print(torch.version.cuda)')"
echo "- OpenCV: $(python -c 'import cv2; print(cv2.__version__)')"
echo ""
echo "実行前に以下の環境変数を設定してください:"
echo ""
echo "import os"
echo "os.environ['QT_QPA_PLATFORM'] = 'offscreen'"
echo "os.environ['LIBGL_ALWAYS_SOFTWARE'] = '1'"
echo ""
echo "準備完了です!"
