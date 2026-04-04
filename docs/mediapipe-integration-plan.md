# FaceFusion × MediaPipe 集成方案

> 目标：将 MediaPipe 的高精度人脸感知能力集成到 FaceFusion 中，同时优化实时流处理管线，使其支持视频会议和直播场景下的实时换脸。

---

## 目录

- [现状分析](#现状分析)
- [技术决策：TFLite vs ONNX](#技术决策tflite-vs-onnx)
- [Phase 1：MediaPipe 作为替代人脸关键点检测器（468 点）](#phase-1mediapipe-作为替代人脸关键点检测器468-点)
- [Phase 2：MediaPipe 作为替代人脸检测器](#phase-2mediapipe-作为替代人脸检测器)
- [Phase 3：利用 Blendshapes 实现实时表情恢复](#phase-3利用-blendshapes-实现实时表情恢复)
- [Phase 4：3D 姿态矩阵改善换脸对齐](#phase-43d-姿态矩阵改善换脸对齐)
- [Phase 5：优化实时流处理管线](#phase-5优化实时流处理管线)
- [文件变更清单](#文件变更清单)
- [实施顺序与依赖关系](#实施顺序与依赖关系)
- [性能预估](#性能预估)

---

## 现状分析

### FaceFusion 已有的实时能力

FaceFusion 并非纯离线工具，**已具备实时流处理基础设施**：

| 模块 | 文件 | 功能 |
|------|------|------|
| 摄像头管理 | `facefusion/camera_manager.py` | 本地/远程摄像头采集（OpenCV VideoCapture） |
| 流处理器 | `facefusion/streamer.py` | 多线程帧处理、UDP/V4L2 输出 |
| Webcam UI | `facefusion/uis/components/webcam.py` | Gradio 网页实时预览 |

流模式支持三种输出：`inline`（内存直出）、`udp`（UDP 推流 localhost:27000）、`v4l2`（虚拟摄像头）。

### 当前瓶颈

| 瓶颈 | 原因 |
|------|------|
| 关键点精度不足 | 仅 5/68 点，大角度侧脸对齐差 |
| 每帧全量检测 | `detect_faces` 无帧间追踪，缓存命中率低 |
| 表情恢复不可用于流模式 | `expression_restorer` 显式禁用 stream 模式（3 个重型模型推理） |
| 无 3D 姿态补偿 | 2D 仿射变换无法处理透视畸变 |
| 无帧丢弃机制 | 处理跟不上采集时队列堆积 |

### MediaPipe 能提供什么

| 能力 | 规格 | 延迟（CPU） |
|------|------|------------|
| Face Mesh | 468 点 3D 关键点（478 含虹膜） | ~10-20ms/帧 |
| Face Detection | 6 关键点，短距/全距两种模型 | ~5ms/帧 |
| Blendshapes | 52 个表情系数（眨眼、微笑、张嘴等） | 含在 Face Mesh 中 |
| 3D 姿态矩阵 | 4×4 变换矩阵（旋转+平移+缩放） | 含在 Face Mesh 中 |
| 虹膜追踪 | 点 468-477 | 含在 Face Mesh 中 |
| 帧间追踪 | VIDEO/LIVE_STREAM 模式内建时序平滑 | 自动 |

---

## 技术决策：TFLite vs ONNX

**方案 A**：直接使用 MediaPipe Python API（`pip install mediapipe`）
**方案 B**：将 TFLite 模型转换为 ONNX，接入 `inference_manager.py`

### 结论：选方案 A

理由：

1. **模型不可拆分** — MediaPipe 的 `.task` 文件是多模型捆绑包（检测器 + Mesh + Blendshape 分类器 + 姿态估计器），无法简单拆分转 ONNX
2. **维护成本** — 每次 MediaPipe 上游更新都要重新转换，部分 TFLite 算子没有 ONNX 对应
3. **实时优化** — MediaPipe LIVE_STREAM 模式支持异步回调和自动丢帧，这正是流处理需要的
4. **性能已验证** — MediaPipe 在手机端都能跑 30fps，无需额外优化

代价是增加一个可选依赖（`mediapipe>=0.10.0`，约 30MB），通过条件导入处理：

```python
# facefusion/mediapipe_manager.py
try:
    import mediapipe
    HAS_MEDIAPIPE = True
except ImportError:
    HAS_MEDIAPIPE = False
```

用户不使用 MediaPipe 功能时无需安装。

---

## Phase 1：MediaPipe 作为替代人脸关键点检测器（468 点）

### 目标

让用户通过 `--face-landmarker-model mediapipe` 选择 MediaPipe 468 点关键点检测，替代现有的 2dfan4/peppa_wutz（68 点）。

### 1.1 类型系统变更

**文件：`facefusion/types.py`**

```python
# 新增类型
FaceLandmark468 : TypeAlias = NDArray[Any]      # shape (468, 3) 含 z 深度
FaceBlendshapes : TypeAlias = NDArray[Any]       # shape (52,) float32
FacePoseMatrix : TypeAlias = NDArray[Any]        # shape (4, 4) float32

# 扩展 FaceLandmarkSet，新增 '468' 键
FaceLandmarkSet = TypedDict('FaceLandmarkSet',
{
    '5' : FaceLandmark5,
    '5/68' : FaceLandmark5,
    '68' : FaceLandmark68,
    '68/5' : FaceLandmark68,
    '468' : FaceLandmark468      # 新增，非 MediaPipe 时为 None
})

# 扩展 FaceLandmarkerModel
FaceLandmarkerModel = Literal['many', '2dfan4', 'peppa_wutz', 'mediapipe']

# 扩展 Face namedtuple（使用 defaults 保持向后兼容）
Face = namedtuple('Face',
[
    'bounding_box', 'score_set', 'landmark_set', 'angle',
    'embedding', 'embedding_norm', 'gender', 'age', 'race',
    'blendshapes',      # 新增 - Optional[FaceBlendshapes]
    'pose_matrix'       # 新增 - Optional[FacePoseMatrix]
], defaults=[None, None])
```

**向后兼容**：`defaults=[None, None]` 确保现有代码用 9 个位置参数构造 `Face` 时不报错。

### 1.2 新建 MediaPipe 管理器

**新文件：`facefusion/mediapipe_manager.py`**

职责：管理 MediaPipe 实例生命周期，类似 `inference_manager.py` 对 ONNX 的管理。

```python
# 核心接口
def get_face_landmarker(running_mode: str = 'image') -> FaceLandmarker
def clear_face_landmarker() -> None
def get_face_detector() -> FaceDetector           # Phase 2
def clear_face_detector() -> None                  # Phase 2
def create_mp_image(vision_frame: VisionFrame) -> mp.Image
def check_mediapipe_available() -> bool
```

关键设计：
- **会话池** — 按 app_context（cli/ui）缓存 FaceLandmarker 实例
- **模型文件** — `face_landmarker_v2.task` 存储在 `.assets/models/`，复用 `conditional_download_hashes` 下载验证
- **运行模式** — 离线处理用 IMAGE/VIDEO 模式，实时流用 LIVE_STREAM 模式

### 1.3 468→68 和 468→5 关键点映射

**新增映射表**（在 `face_landmarker.py` 或 `face_helper.py` 中定义）：

MediaPipe 468 点使用不同的编号体系，需要映射到标准 68 点（iBUG 格式）：

```python
# 468 -> 68 点映射（iBUG 标准）
MEDIAPIPE_TO_68 = [
    # 下颌轮廓 (17 点)
    162, 234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365,
    # 左眉 (5 点)
    70, 63, 105, 66, 107,
    # 右眉 (5 点)
    336, 296, 334, 293, 300,
    # 鼻梁 (4 点)
    168, 6, 197, 195,
    # 鼻尖 (5 点)
    5, 4, 1, 19, 94,
    # 左眼 (6 点)
    33, 160, 158, 133, 153, 144,
    # 右眼 (6 点)
    362, 385, 387, 263, 373, 380,
    # 外唇 (12 点)
    61, 40, 37, 0, 267, 270, 291, 321, 314, 17, 84, 181,
    # 内唇 (8 点)
    78, 82, 13, 312, 308, 317, 14, 87,
]

# 468 -> 5 点映射（眼中心、鼻尖、嘴角）
MEDIAPIPE_TO_5 = [
    # left_eye_center, right_eye_center, nose_tip, left_mouth, right_mouth
    # 通过左右眼关键点平均计算，或直接取特定索引
]
```

### 1.4 修改 face_landmarker.py

**文件：`facefusion/face_landmarker.py`**

新增 `detect_with_mediapipe` 函数：

```python
def detect_with_mediapipe(
    vision_frame: VisionFrame,
    bounding_box: BoundingBox,
    face_angle: Angle
) -> Tuple[FaceLandmark68, Score, FaceLandmark468, FaceBlendshapes, FacePoseMatrix]:
    """
    1. 裁剪人脸区域（带 padding）
    2. 创建 mediapipe.Image
    3. 调用 face_landmarker.detect()
    4. 提取 468 点归一化坐标 -> 转为像素坐标
    5. 通过 MEDIAPIPE_TO_68 映射得到 68 点
    6. 通过 MEDIAPIPE_TO_5 映射得到 5 点
    7. 返回 68 点 + 置信度 + 468 点 + blendshapes + pose_matrix
    """
```

修改 `detect_face_landmark` 的分发逻辑：

```python
face_landmarker_model = state_manager.get_item('face_landmarker_model')

if face_landmarker_model in ['many', 'mediapipe']:
    result = detect_with_mediapipe(vision_frame, bounding_box, face_angle)
    # ...
```

修改 `create_static_model_set`：新增 `'mediapipe'` 条目（指向 `.task` 文件的哈希和下载 URL）。

修改 `get_inference_pool` / `clear_inference_pool`：当模型为 `'mediapipe'` 时，委托给 `mediapipe_manager` 而非 ONNX。

### 1.5 修改 face_analyser.py

**文件：`facefusion/face_analyser.py`**

在 `create_faces` 中：

```python
# 构建 landmark_set 时包含 468 点
face_landmark_set = {
    '5': face_landmark_5,
    '5/68': face_landmark_5_68,
    '68': face_landmark_68,
    '68/5': face_landmark_68_5,
    '468': face_landmark_468       # 非 mediapipe 时为 None
}

# 构造 Face 时传入新字段
face = Face(
    bounding_box, score_set, face_landmark_set, angle,
    embedding, embedding_norm, gender, age, race,
    blendshapes=blendshapes,       # 非 mediapipe 时为 None
    pose_matrix=pose_matrix        # 非 mediapipe 时为 None
)
```

在 `scale_face` 中：新增对 `'468'` 键的缩放处理。

### 1.6 向后兼容保证

- 所有读取 `landmark_set.get('5')`, `landmark_set.get('68')` 的现有代码**无需修改**
- 468→68 和 468→5 映射确保这些键始终被填充
- `'468'` 键仅在选择 mediapipe 时有值
- 不使用 blendshapes/pose_matrix 的处理器自动忽略这些字段

---

## Phase 2：MediaPipe 作为替代人脸检测器

### 目标

让用户通过 `--face-detector-model mediapipe` 选择 MediaPipe 人脸检测。

### 2.1 类型变更

**文件：`facefusion/types.py`**

```python
FaceDetectorModel = Literal['many', 'retinaface', 'scrfd', 'yolo_face', 'yunet', 'mediapipe']
```

### 2.2 修改 face_detector.py

**文件：`facefusion/face_detector.py`**

新增函数：

```python
def detect_with_mediapipe(
    vision_frame: VisionFrame,
    face_detector_size: str
) -> Tuple[List[BoundingBox], List[Score], List[FaceLandmark5]]:
    """
    1. 使用 mediapipe_manager.get_face_detector()
    2. 转换 VisionFrame -> mediapipe.Image
    3. 调用 detector.detect()
    4. 归一化 bbox -> 像素坐标
    5. 6 关键点 -> 5 点映射
    """
```

**关键点映射问题**：MediaPipe 检测器输出 6 个关键点（右眼、左眼、鼻尖、嘴中心、右耳、左耳），而 FaceFusion 需要 5 个（左眼、右眼、鼻、左嘴角、右嘴角）。

解决方案：
- 眼睛和鼻尖直接映射
- 嘴角从嘴中心点和 bbox 宽度近似推算
- 当同时使用 mediapipe 检测器 + mediapipe 关键点检测器时，5 点可直接从 468 mesh 精确提取

### 2.3 修改 choices.py

**文件：`facefusion/choices.py`**

```python
face_detector_set = {
    # ... 现有条目 ...
    'mediapipe': ['640x640']
}
```

### 2.4 性能优势

MediaPipe 检测器极快（CPU ~5ms），对流模式延迟改善显著。

---

## Phase 3：利用 Blendshapes 实现实时表情恢复

### 目标

用 MediaPipe 的 52 个 blendshapes 替代 expression_restorer 中昂贵的 motion_extractor 推理，使表情恢复可用于流模式。

### 3.1 问题分析

当前 expression_restorer 的每帧开销：

```
forward_extract_feature()   → ~30ms    特征提取（可缓存）
forward_extract_motion() ×2 → ~60ms    运动提取（目标帧+处理帧各一次）
forward_generate_frame()    → ~50ms    生成帧
────────────────────────────────────
总计                         ~140ms/帧  → 仅 7fps，无法实时
```

### 3.2 优化策略

```
                      现有方案                    优化方案
──────────────────────────────────────────────────────────────
特征提取    forward_extract_feature (30ms)    缓存（同一人脸跨帧复用）→ ~0ms
运动提取    forward_extract_motion ×2 (60ms)  MediaPipe blendshapes (15ms) + 映射 (<1ms)
帧生成      forward_generate_frame (50ms)     不变 (50ms)
──────────────────────────────────────────────────────────────
总计        ~140ms (7fps)                     ~65ms (15fps)
```

### 3.3 Blendshape → LivePortrait 表情映射

**新文件：`facefusion/processors/blendshape_mapper.py`**

MediaPipe 有 52 个 blendshapes，LivePortrait 有 21×3 个表情系数。需要建立映射关系：

```python
# 语义对应关系（部分）
# MediaPipe                    → LivePortrait 系数索引
# EYE_BLINK_LEFT/RIGHT         → [1, 2]     上脸-眼部
# JAW_OPEN                     → [3, 7]     下脸
# MOUTH_SMILE_LEFT/RIGHT       → [19, 20]   嘴部
# BROW_DOWN_LEFT/RIGHT         → [10, 11]   眉部
# BROW_INNER_UP                → [12]       眉部

def map_blendshapes_to_expression(blendshapes: FaceBlendshapes) -> NDArray:
    """52 个 blendshapes -> 21×3 个 LivePortrait 表情系数"""
    # 方案 A：手工线性映射矩阵 (52 × 63)
    # 方案 B：训练小型 MLP 回归模型（在配对数据上）
```

推荐**先用方案 A 手工映射**快速验证，效果不理想再训练方案 B。

### 3.4 修改 expression_restorer

**文件：`facefusion/processors/modules/expression_restorer/core.py`**

1. 取消流模式的硬性禁用：

```python
# 修改前
if mode == 'stream':
    logger.error(...)
    return False

# 修改后
if mode == 'stream' and state_manager.get_item('face_landmarker_model') != 'mediapipe':
    logger.error('实时表情恢复需要 mediapipe 关键点检测器', __name__)
    return False
```

2. 新增轻量化恢复路径：

```python
def apply_restore_realtime(target_crop_vision_frame, temp_crop_vision_frame, factor, face):
    """使用缓存特征 + MediaPipe blendshapes 的轻量路径"""
    # 缓存特征卷（同一人脸跨帧不变）
    feature_volume = get_cached_feature_volume(temp_crop_vision_frame, face)
    
    # 从 Face 对象直接获取 blendshapes（MediaPipe 已计算好）
    target_expression = map_blendshapes_to_expression(face.blendshapes)
    
    # 仍需一次 motion_extractor 获取姿态（pitch, yaw, roll, scale, translation）
    # 但表情系数部分用 blendshapes 替代
    pitch, yaw, roll, scale, translation, _, motion_points = forward_extract_motion(...)
    
    # 生成帧
    crop_vision_frame = forward_generate_frame(feature_volume, target_motion_points, temp_motion_points)
    return crop_vision_frame
```

3. 特征缓存机制：

```python
_cached_feature_volume = None
_cached_face_hash = None

def get_cached_feature_volume(crop_vision_frame, face):
    global _cached_feature_volume, _cached_face_hash
    face_hash = hash(face.embedding.tobytes())
    if face_hash != _cached_face_hash:
        _cached_feature_volume = forward_extract_feature(crop_vision_frame)
        _cached_face_hash = face_hash
    return _cached_feature_volume
```

---

## Phase 4：3D 姿态矩阵改善换脸对齐

### 目标

利用 MediaPipe 的 4×4 姿态矩阵改善大角度换脸效果，消除"纸片感"。

### 4.1 问题分析

当前对齐方式：

```python
# face_helper.py 中的 warp_face_by_face_landmark_5
affine_matrix = cv2.estimateAffinePartial2D(face_landmark_5, warp_template)
# → 仅 2D 仿射变换（旋转+缩放+平移），无透视
```

大角度侧脸时，2D 仿射无法处理透视畸变，导致换脸后面部看起来扁平。

### 4.2 新建 3D 姿态工具

**新文件：`facefusion/face_pose.py`**

```python
def decompose_pose_matrix(pose_matrix: FacePoseMatrix) -> Tuple[float, float, float, NDArray]:
    """从 4×4 矩阵分解出 pitch, yaw, roll（角度）和 translation"""

def estimate_improved_affine(
    face_landmark_5: FaceLandmark5,
    pose_matrix: FacePoseMatrix,
    warp_template: WarpTemplate,
    crop_size: Size
) -> Matrix:
    """计算透视感知的仿射变换"""

def compensate_perspective(
    crop_vision_frame: VisionFrame,
    source_pose_matrix: FacePoseMatrix,
    target_pose_matrix: FacePoseMatrix
) -> VisionFrame:
    """源脸和目标脸之间的透视补偿"""
```

### 4.3 修改 face_swapper

**文件：`facefusion/processors/modules/face_swapper/core.py`**

在 `swap_face` 中，当 `target_face.pose_matrix is not None` 时：

```python
def swap_face(source_face, target_face, temp_vision_frame):
    # ... 现有裁剪和对齐 ...
    
    if target_face.pose_matrix is not None:
        yaw = decompose_pose_matrix(target_face.pose_matrix)[1]  # 偏航角
        if abs(yaw) > 30:  # 大角度时启用透视补偿
            crop_vision_frame = compensate_perspective(
                crop_vision_frame,
                source_face.pose_matrix,
                target_face.pose_matrix
            )
    
    # ... 后续换脸推理和贴回 ...
```

### 4.4 向后兼容

- 仅当 `pose_matrix is not None` 时启用（即仅 mediapipe 模式）
- 现有 2D 仿射路径作为默认回退
- 大角度阈值可配置

---

## Phase 5：优化实时流处理管线

### 目标

使现有流处理管线达到 720p@25-30fps 的会议/直播可用水平。

### 5.1 帧间追踪（减少检测频率）

**新文件：`facefusion/face_tracker.py`**

```python
class FaceTracker:
    def __init__(self, detect_interval: int = 5):
        self.detect_interval = detect_interval
        self.frame_count = 0
        self.tracked_faces: List[Face] = []
    
    def update(self, vision_frame: VisionFrame) -> List[Face]:
        if self.frame_count % self.detect_interval == 0:
            # 每 N 帧全量检测
            self.tracked_faces = full_detection(vision_frame)
        else:
            # 中间帧：在已知 bbox 扩展区域内做轻量关键点检测
            self.tracked_faces = track_existing(vision_frame, self.tracked_faces)
        self.frame_count += 1
        return self.tracked_faces
```

**与 MediaPipe 的协同**：MediaPipe VIDEO/LIVE_STREAM 模式内建帧间时序平滑，`detect_for_video()` 传入递增时间戳时自动利用前帧信息。

### 5.2 帧丢弃机制

**文件：`facefusion/streamer.py`** — 修改 `multi_process_capture`

当前问题：每帧都提交处理，处理速度跟不上时队列堆积。

```python
# 优化方案：最新帧策略
def multi_process_capture(camera_capture, camera_fps):
    latest_frame = None
    frame_lock = threading.Lock()
    
    # 采集线程：始终更新 latest_frame（覆盖旧帧）
    def capture_loop():
        nonlocal latest_frame
        while running:
            ret, frame = camera_capture.read()
            if ret:
                with frame_lock:
                    latest_frame = frame
    
    # 处理线程：取最新帧处理
    def process_loop():
        while running:
            with frame_lock:
                frame = latest_frame
            if frame is not None:
                processed = process_stream_frame(frame)
                yield processed
```

### 5.3 异步 MediaPipe 管线

**文件：`facefusion/mediapipe_manager.py`** — 新增

```python
def create_live_stream_landmarker(callback):
    """创建 LIVE_STREAM 模式的 landmarker，结果通过回调异步返回"""
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionTaskRunningMode.LIVE_STREAM,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        result_callback=callback,
        num_faces=1
    )
    return FaceLandmarker.create_from_options(options)
```

MediaPipe 的 LIVE_STREAM 模式：
- 异步提交帧，结果通过回调返回
- 如果新帧到达时旧帧还没处理完，旧帧自动丢弃
- 这正是流处理需要的行为

### 5.4 处理分辨率降级

**文件：`facefusion/streamer.py`** — 修改 `process_stream_frame`

```python
def process_stream_frame(target_vision_frame, process_scale=0.5):
    # 降分辨率处理（720p -> 360p）
    h, w = target_vision_frame.shape[:2]
    small = cv2.resize(target_vision_frame, (int(w * process_scale), int(h * process_scale)))
    
    # 在低分辨率上执行检测+换脸
    processed_small = run_processors(small)
    
    # 放大回原始分辨率
    processed = cv2.resize(processed_small, (w, h))
    return processed
```

720p→360p 处理可将检测+换脸时间减半，配合 face_enhancer 可部分弥补画质损失。

### 5.5 内容分析节流

**文件：`facefusion/streamer.py`**

当前 `analyse_stream`（NSFW 检测等）每帧执行，改为每秒检测一次：

```python
if frame_count % int(camera_fps) == 0:  # 每秒一次
    if analyse_stream(capture_vision_frame, camera_fps):
        camera_capture.release()
frame_count += 1
```

### 5.6 新增流配置项

**文件：`facefusion/types.py`**

```python
# 新增 state 键
'stream_detect_interval': int       # 帧间检测间隔（默认 5）
'stream_process_scale': float       # 处理缩放比（默认 0.5）
'stream_frame_drop': bool           # 启用帧丢弃（默认 True）
```

**文件：`facefusion/uis/components/webcam.py`**

新增对应的 UI 控件（滑块/开关）。

---

## 文件变更清单

### 新建文件

| 文件 | Phase | 用途 |
|------|-------|------|
| `facefusion/mediapipe_manager.py` | 1 | MediaPipe 实例生命周期管理 |
| `facefusion/processors/blendshape_mapper.py` | 3 | Blendshapes → LivePortrait 表情映射 |
| `facefusion/face_pose.py` | 4 | 3D 姿态矩阵工具 |
| `facefusion/face_tracker.py` | 5 | 帧间人脸追踪 |

### 修改文件

| 文件 | Phase | 变更内容 |
|------|-------|---------|
| `facefusion/types.py` | 1,2,4,5 | 新类型、扩展 Literal 和 TypedDict |
| `facefusion/face_landmarker.py` | 1 | `detect_with_mediapipe`、模型集、推理池 |
| `facefusion/face_detector.py` | 2 | `detect_with_mediapipe`、模型集 |
| `facefusion/face_analyser.py` | 1 | 填充新 Face 字段、缩放 468 点 |
| `facefusion/face_helper.py` | 4 | 3D 感知对齐、增强角度估计 |
| `facefusion/choices.py` | 2 | mediapipe 检测器尺寸配置 |
| `facefusion/streamer.py` | 5 | 帧丢弃、异步管线、分辨率降级、分析节流 |
| `facefusion/processors/modules/expression_restorer/core.py` | 3 | 启用流模式、混合 blendshape 路径、特征缓存 |
| `facefusion/processors/modules/face_swapper/core.py` | 4 | 透视感知对齐 |
| `facefusion/processors/modules/face_debugger/core.py` | 1 | 可视化 468 点（调试用） |
| `facefusion/uis/components/webcam.py` | 5 | 新增流配置 UI 控件 |
| `requirements.txt` | 1 | 可选依赖 `mediapipe>=0.10.0` |

---

## 实施顺序与依赖关系

```
Phase 1 ─── 关键点检测器（基础，其他所有 Phase 依赖）
  │
  ├── Phase 2 ─── 人脸检测器（独立于 3/4/5，可并行）
  │
  ├── Phase 3 ─── 实时表情恢复（依赖 Phase 1 的 blendshapes）
  │
  ├── Phase 4 ─── 3D 姿态对齐（依赖 Phase 1 的 pose_matrix）
  │
  └── Phase 5 ─── 流管线优化
        │
        ├── 5.2 帧丢弃        ← 无依赖，可最先做
        ├── 5.5 分析节流       ← 无依赖，可最先做
        ├── 5.4 分辨率降级     ← 无依赖，可最先做
        ├── 5.1 帧间追踪       ← 依赖 Phase 1（MediaPipe VIDEO 模式）
        └── 5.3 异步管线       ← 依赖 Phase 1（MediaPipe LIVE_STREAM 模式）
```

**推荐实施顺序**：

1. **先做 Phase 5.2 + 5.4 + 5.5**（纯优化，无外部依赖，立即见效）
2. **再做 Phase 1**（核心集成，后续 Phase 的基础）
3. **Phase 2 和 Phase 3 并行开发**
4. **最后做 Phase 4**（锦上添花，依赖前面所有工作验证）
5. **Phase 5.1 + 5.3 穿插在 Phase 1 之后**

---

## 性能预估

### 离线视频换脸（改善质量）

| 环节 | 现有方案 | 集成后 |
|------|---------|--------|
| 关键点精度 | 68 点 2D | 468 点 3D |
| 大角度对齐 | 2D 仿射（有畸变） | 3D 透视补偿 |
| 遮罩精度 | box/occlusion/region | + 468 点精细轮廓 |
| 处理速度 | 不变 | 略慢（多一次 MediaPipe 推理） |
| **整体效果** | **良好** | **显著提升，尤其侧脸** |

### 实时流换脸（720p 目标）

| 环节 | 优化前 | 优化后 | 节省 |
|------|--------|--------|------|
| 人脸检测 | ONNX 每帧 ~30ms | MediaPipe 每 5 帧 ~5ms，均摊 ~1ms | ~29ms |
| 关键点检测 | ONNX 68 点 ~20ms | MediaPipe 468 点 ~15ms | ~5ms |
| 换脸推理 | ~40ms (720p) | ~25ms (360p 处理+放大) | ~15ms |
| 表情恢复 | 不可用 (140ms) | ~65ms（缓存+blendshapes） | 可用 |
| 内容分析 | 每帧 ~10ms | 每秒一次，均摊 ~0.3ms | ~10ms |
| **总计** | **~100ms+ (≤10fps)** | **~42ms (~24fps)** | |

**结论**：通过全部 5 个 Phase 的优化，实时流换脸在 RTX 3060+ 级别 GPU 上可达到 **720p@20-25fps**，满足视频会议和直播的基本需求。配合帧丢弃和时序平滑，用户感知流畅度可接近 30fps。

---

## 会议/直播场景使用方式

集成完成后，用户的使用流程：

```bash
# 方式一：Web UI
python facefusion.py run
# → 选择 Webcam 标签页
# → 上传源脸照片
# → 选择 face_landmarker_model = mediapipe
# → 选择 processors = face_swapper (+ face_enhancer 可选)
# → 选择摄像头、分辨率、帧率
# → 输出模式选 v4l2（虚拟摄像头）
# → 点击 Start
# → 在腾讯会议/Zoom/OBS 中选择虚拟摄像头作为视频源

# 方式二：命令行
python facefusion.py run-stream \
    --source-paths trump.jpg \
    --processors face_swapper \
    --face-landmarker-model mediapipe \
    --face-detector-model mediapipe \
    --stream-mode v4l2 \
    --stream-resolution 1280x720 \
    --stream-fps 30
```
