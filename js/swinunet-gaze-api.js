/**
 * SwinUNet Gaze Estimation API
 * 
 * 用于在浏览器中进行实时视线估计的 JavaScript API。
 * 基于 TensorFlow.js 和 MediaPipe Face Mesh 实现。
 * 
 * @version 1.0.0
 * @author SwinUNet-VOG Project
 * @license CC BY-NC-SA 4.0
 */

class SwinUNetGazeAPI {
    constructor() {
        this.model = null;
        this.faceMesh = null;
        this.isInitialized = false;
        
        // 模型输入尺寸（与训练时一致）
        this.inputSize = { width: 60, height: 36 };
        this.targetSize = { width: 60, height: 36 };
        
        // MediaPipe Face Mesh 关键点索引
        // 参考：https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png
        this.LEFT_EYE_OUTER = 263;
        this.LEFT_EYE_INNER = 362;
        this.LEFT_EYE_UPPER = 386;
        this.RIGHT_EYE_OUTER = 33;
        this.RIGHT_EYE_INNER = 133;
        this.RIGHT_EYE_UPPER = 159;
        
        // 历史记录（用于可视化）
        this.gazeHistory = [];
        this.timeHistory = [];
    }
    
    /**
     * 初始化模型和 MediaPipe
     * 
     * @param {string} modelPath - TensorFlow.js 模型路径（model.json）
     * @returns {Promise<void>}
     */
    async initialize(modelPath = 'models/tfjs_model/model.json') {
        console.log('[SwinUNet] Initializing...');
        
        try {
            // 加载 TensorFlow.js 模型
            console.log('[SwinUNet] Loading model from:', modelPath);
            this.model = await tf.loadGraphModel(modelPath);
            console.log('[SwinUNet] Model loaded');
            
            // 初始化 MediaPipe Face Mesh
            console.log('[SwinUNet] Initializing MediaPipe Face Mesh...');
            this.faceMesh = new FaceMesh({
                locateFile: (file) => {
                    return `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`;
                }
            });
            
            this.faceMesh.setOptions({
                maxNumFaces: 1,
                refineLandmarks: true,
                minDetectionConfidence: 0.5,
                minTrackingConfidence: 0.5
            });
            
            console.log('[SwinUNet] MediaPipe initialized');
            
            this.isInitialized = true;
            console.log('[SwinUNet] Ready');
            
        } catch (error) {
            console.error('[SwinUNet] Initialization failed:', error);
            throw error;
        }
    }
    
    /**
     * 从输入源提取眼部 ROI
     * 
     * 使用 MediaPipe 检测人脸关键点，然后通过仿射变换提取眼部区域。
     * 这一步骤对应 Python 代码中的 MediaPipeEyeNormalizer.extract() 方法。
     * 
     * @param {HTMLVideoElement|HTMLImageElement|HTMLCanvasElement} source - 输入源
     * @param {string} eye - 选择眼睛：'left' 或 'right'
     * @returns {Promise<tf.Tensor|null>} 眼部图像张量 (1, 3, 36, 60) 或 null
     */
    async extractEyeROI(source, eye = 'left') {
        if (!this.faceMesh) {
            throw new Error('[SwinUNet] MediaPipe not initialized');
        }
        
        // MediaPipe 人脸检测
        const results = await this.faceMesh.send({ image: source });
        
        if (!results || !results.multiFaceLandmarks || results.multiFaceLandmarks.length === 0) {
            return null;  // 未检测到人脸
        }
        
        const landmarks = results.multiFaceLandmarks[0];
        
        // 根据选择的眼睛获取关键点索引
        let outerIdx, innerIdx, upperIdx;
        if (eye === 'right') {
            outerIdx = this.RIGHT_EYE_OUTER;
            innerIdx = this.RIGHT_EYE_INNER;
            upperIdx = this.RIGHT_EYE_UPPER;
        } else {
            outerIdx = this.LEFT_EYE_OUTER;
            innerIdx = this.LEFT_EYE_INNER;
            upperIdx = this.LEFT_EYE_UPPER;
        }
        
        const outerCorner = landmarks[outerIdx];
        const innerCorner = landmarks[innerIdx];
        const upperCenter = landmarks[upperIdx];
        
        // MediaPipe 返回归一化坐标 (0-1)，转换为像素坐标
        const width = source.videoWidth || source.width;
        const height = source.videoHeight || source.height;
        
        const srcPoints = [
            [outerCorner.x * width, outerCorner.y * height],
            [innerCorner.x * width, innerCorner.y * height],
            [upperCenter.x * width, upperCenter.y * height]
        ];
        
        // 目标点：标准化眼部位置
        // 这些坐标与 Python 代码中的 target_points 对应
        const padding = 0.15;
        const dstPoints = [
            [this.targetSize.width * padding, this.targetSize.height * 0.55],
            [this.targetSize.width * (1 - padding), this.targetSize.height * 0.55],
            [this.targetSize.width * 0.5, this.targetSize.height * 0.25]
        ];
        
        // 仿射变换并裁剪眼部区域
        const eyeROI = await this.warpAffine(source, srcPoints, dstPoints, this.targetSize);
        
        // 右眼需要水平翻转（与 Python 代码一致）
        if (eye === 'right') {
            return tf.image.flipLeftRight(eyeROI);
        }
        
        return eyeROI;
    }
    
    /**
     * 仿射变换
     * 
     * 简化实现：使用边界框裁剪代替完整的仿射变换矩阵。
     * 完整实现需要计算仿射变换矩阵并应用到每个像素。
     * 
     * @param {HTMLVideoElement|HTMLImageElement|HTMLCanvasElement} source - 输入源
     * @param {Array} srcPoints - 源点 [[x1,y1], [x2,y2], [x3,y3]]
     * @param {Array} dstPoints - 目标点 [[x1,y1], [x2,y2], [x3,y3]]
     * @param {Object} targetSize - 目标尺寸 {width, height}
     * @returns {Promise<tf.Tensor>} 变换后的图像张量 (1, 3, H, W)
     */
    async warpAffine(source, srcPoints, dstPoints, targetSize) {
        // 将源图像转换为张量
        let imageTensor = tf.browser.fromPixels(source);
        
        // 计算眼部区域的边界框
        const minX = Math.min(...srcPoints.map(p => p[0]));
        const maxX = Math.max(...srcPoints.map(p => p[0]));
        const minY = Math.min(...srcPoints.map(p => p[1]));
        const maxY = Math.max(...srcPoints.map(p => p[1]));
        
        const boxWidth = maxX - minX;
        const boxHeight = maxY - minY;
        
        // 添加 padding
        const pad = 0.2;
        const x1 = Math.max(0, minX - boxWidth * pad);
        const y1 = Math.max(0, minY - boxHeight * pad);
        const x2 = Math.min(source.width || source.videoWidth, maxX + boxWidth * pad);
        const y2 = Math.min(source.height || source.videoHeight, maxY + boxHeight * pad);
        
        // 裁剪并调整大小
        const cropped = tf.image.cropAndResize(
            imageTensor.expandDims(0),
            [[y1 / imageTensor.shape[0], x1 / imageTensor.shape[1], 
              y2 / imageTensor.shape[0], x2 / imageTensor.shape[1]]],
            [0],
            [targetSize.height, targetSize.width]
        );
        
        // 归一化到 [0, 1]
        const normalized = cropped.div(255.0);
        
        // 转换为 NCHW 格式（与 PyTorch 一致）
        const transposed = normalized.transpose([0, 3, 1, 2]);
        
        // 释放中间张量
        imageTensor.dispose();
        cropped.dispose();
        normalized.dispose();
        
        return transposed;
    }
    
    /**
     * 预测视线方向
     * 
     * @param {tf.Tensor} eyeROI - 眼部 ROI 张量 (1, 3, 36, 60)
     * @returns {Promise<Object>} { vector: [x,y,z], pitch: degrees, yaw: degrees }
     */
    async predictGaze(eyeROI) {
        if (!this.model) {
            throw new Error('[SwinUNet] Model not loaded');
        }
        
        // 模型推理
        const output = await this.model.predict(eyeROI);
        const gazeVector = await output.data();
        
        // 归一化为单位向量
        const norm = Math.sqrt(
            gazeVector[0] ** 2 + 
            gazeVector[1] ** 2 + 
            gazeVector[2] ** 2
        );
        
        const normalizedVector = [
            gazeVector[0] / norm,
            gazeVector[1] / norm,
            gazeVector[2] / norm
        ];
        
        // 转换为角度
        const angles = this.vectorToAngles(normalizedVector);
        
        output.dispose();
        
        return {
            vector: normalizedVector,
            pitch: angles.pitch,
            yaw: angles.yaw
        };
    }
    
    /**
     * 3D 向量转换为 Pitch/Yaw 角度
     * 
     * 使用 MPIIGaze 标准转换公式：
     * - pitch = arcsin(-y)
     * - yaw = arctan2(-x, -z)
     * 
     * 负号是因为 MPIIGaze 坐标系中，相机朝向为 (0, 0, -1)。
     * 
     * @param {Array} vector - 3D 向量 [x, y, z]
     * @returns {Object} { pitch: degrees, yaw: degrees }
     */
    vectorToAngles(vector) {
        const [x, y, z] = vector;
        
        // MPIIGaze 标准转换
        const pitch = Math.asin(-y) * 180 / Math.PI;
        const yaw = Math.atan2(-x, -z) * 180 / Math.PI;
        
        return { pitch, yaw };
    }
    
    /**
     * 处理视频帧（完整流程）
     * 
     * 这是最常用的方法，封装了完整的处理流程：
     * 1. 提取眼部 ROI
     * 2. 预测视线
     * 3. 记录历史
     * 
     * @param {HTMLVideoElement|HTMLImageElement} source - 输入源
     * @param {string} eye - 'left' 或 'right'
     * @returns {Promise<Object|null>} 视线预测结果或 null
     */
    async processFrame(source, eye = 'left') {
        try {
            // 提取眼部 ROI
            const eyeROI = await this.extractEyeROI(source, eye);
            if (!eyeROI) {
                return null;
            }
            
            // 预测视线
            const gaze = await this.predictGaze(eyeROI);
            
            // 记录历史
            const timestamp = Date.now();
            this.gazeHistory.push(gaze);
            this.timeHistory.push(timestamp);
            
            // 限制历史记录长度（防止内存溢出）
            if (this.gazeHistory.length > 300) {
                this.gazeHistory.shift();
                this.timeHistory.shift();
            }
            
            eyeROI.dispose();
            
            return gaze;
            
        } catch (error) {
            console.error('[SwinUNet] Frame processing error:', error);
            return null;
        }
    }
    
    /**
     * 获取视线历史记录
     * 
     * @returns {Object} { gaze: Array, time: Array }
     */
    getHistory() {
        return {
            gaze: this.gazeHistory,
            time: this.timeHistory
        };
    }
    
    /**
     * 清空历史记录
     */
    clearHistory() {
        this.gazeHistory = [];
        this.timeHistory = [];
    }
    
    /**
     * 释放资源
     * 
     * 在不再使用 API 时调用，释放模型和 MediaPipe 占用的资源。
     */
    dispose() {
        if (this.model) {
            this.model.dispose();
        }
        if (this.faceMesh) {
            this.faceMesh.close();
        }
        this.isInitialized = false;
        console.log('[SwinUNet] Disposed');
    }
}

// 导出（支持 Node.js 和浏览器）
if (typeof module !== 'undefined' && module.exports) {
    module.exports = SwinUNetGazeAPI;
}
