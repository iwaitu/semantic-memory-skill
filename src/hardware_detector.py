"""
硬件检测模块 - 自动检测最优加速方案
支持：Apple CoreML, NVIDIA CUDA/TensorRT, CPU ONNX
"""
import platform
import subprocess
from enum import Enum
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)


class AcceleratorType(Enum):
    """加速方案类型"""
    APPLE_COREML = "coreml"       # Apple Silicon (最快 ~1-2ms)
    NVIDIA_TENSORRT = "tensorrt"  # NVIDIA TensorRT (~1-2ms)
    NVIDIA_CUDA = "cuda"          # NVIDIA CUDA (~2-3ms)
    CPU_AVX512 = "cpu_avx512"     # CPU with AVX-512 (~3-4ms)
    CPU_ONNX = "cpu"              # CPU fallback (~5ms)


class HardwareDetector:
    """硬件检测器"""
    
    @staticmethod
    def detect() -> AcceleratorType:
        """
        自动检测硬件并返回最优加速方案
        
        检测顺序：
        1. Apple Silicon (CoreML)
        2. NVIDIA GPU + TensorRT
        3. NVIDIA GPU + CUDA
        4. CPU with AVX-512
        5. CPU fallback
        """
        system = platform.system()
        logger.info(f"检测平台：{system}")
        
        # Apple Silicon 检测
        if system == "Darwin":
            if HardwareDetector._is_apple_silicon():
                logger.info("✅ 检测到 Apple Silicon")
                return AcceleratorType.APPLE_COREML
            logger.info("ℹ️ Intel Mac，使用 CPU 模式")
            return AcceleratorType.CPU_ONNX
        
        # Linux/Windows - NVIDIA GPU 检测
        if HardwareDetector._has_nvidia_gpu():
            if HardwareDetector._has_tensorrt():
                logger.info("✅ 检测到 NVIDIA GPU + TensorRT")
                return AcceleratorType.NVIDIA_TENSORRT
            logger.info("✅ 检测到 NVIDIA GPU (CUDA)")
            return AcceleratorType.NVIDIA_CUDA
        
        # CPU 检测
        if HardwareDetector._has_avx512():
            logger.info("ℹ️ 检测到 AVX-512 支持")
            return AcceleratorType.CPU_AVX512
        
        logger.info("ℹ️ 使用标准 CPU 模式")
        return AcceleratorType.CPU_ONNX
    
    @staticmethod
    def _is_apple_silicon() -> bool:
        """检测是否为 Apple Silicon (M1/M2/M3)"""
        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=5
            )
            return "apple" in result.stdout.lower()
        except Exception as e:
            logger.warning(f"Apple Silicon 检测失败：{e}")
            return False
    
    @staticmethod
    def _has_nvidia_gpu() -> bool:
        """检测 NVIDIA GPU"""
        try:
            # 方法 1: nvidia-smi
            result = subprocess.run(
                ["nvidia-smi"],
                capture_output=True, timeout=5
            )
            if result.returncode == 0:
                return True
            
            # 方法 2: PyTorch CUDA
            try:
                import torch
                if torch.cuda.is_available():
                    return True
            except ImportError:
                pass
            
            return False
        except Exception as e:
            logger.warning(f"NVIDIA GPU 检测失败：{e}")
            return False
    
    @staticmethod
    def _has_tensorrt() -> bool:
        """检测 TensorRT 支持"""
        try:
            import tensorrt
            logger.info(f"TensorRT 版本：{tensorrt.__version__}")
            return True
        except ImportError:
            return False
    
    @staticmethod
    def _has_avx512() -> bool:
        """检测 AVX-512 支持"""
        try:
            import cpuinfo
            info = cpuinfo.get_cpu_info()
            flags = info.get('flags', [])
            return 'avx512f' in flags or 'avx512' in ' '.join(flags)
        except ImportError:
            # Fallback: 检查 /proc/cpuinfo
            try:
                with open('/proc/cpuinfo', 'r') as f:
                    content = f.read()
                    return 'avx512' in content
            except:
                return False
    
    @staticmethod
    def get_install_packages(accel_type: AcceleratorType) -> List[str]:
        """返回对应加速方案的 pip 安装包列表"""
        package_map = {
            AcceleratorType.APPLE_COREML: [
                "onnxruntime-coreml>=1.16.0",
                "optimum[coreml]>=1.17.0",
                "coremltools>=7.0",
                "sentence-transformers>=2.2.0"
            ],
            AcceleratorType.NVIDIA_TENSORRT: [
                "onnxruntime-gpu>=1.16.0",
                "tensorrt>=8.6.0",
                "optimum[tensorrt]>=1.17.0",
                "sentence-transformers>=2.2.0"
            ],
            AcceleratorType.NVIDIA_CUDA: [
                "onnxruntime-gpu>=1.16.0",
                "optimum[onnxruntime-gpu]>=1.17.0",
                "sentence-transformers>=2.2.0"
            ],
            AcceleratorType.CPU_AVX512: [
                "onnxruntime>=1.16.0",
                "optimum[onnxruntime]>=1.17.0",
                "sentence-transformers>=2.2.0"
            ],
            AcceleratorType.CPU_ONNX: [
                "onnxruntime>=1.16.0",
                "optimum[onnxruntime]>=1.17.0",
                "sentence-transformers>=2.2.0"
            ]
        }
        return package_map.get(accel_type, package_map[AcceleratorType.CPU_ONNX])
    
    @staticmethod
    def get_model_export_format(accel_type: AcceleratorType) -> str:
        """返回最优模型导出格式"""
        format_map = {
            AcceleratorType.APPLE_COREML: "coreml",
            AcceleratorType.NVIDIA_TENSORRT: "engine",
            AcceleratorType.NVIDIA_CUDA: "onnx",
            AcceleratorType.CPU_AVX512: "onnx",
            AcceleratorType.CPU_ONNX: "onnx"
        }
        return format_map.get(accel_type, "onnx")
    
    @staticmethod
    def get_accelerator_info(accel_type: AcceleratorType) -> dict:
        """返回加速方案详细信息"""
        info_map = {
            AcceleratorType.APPLE_COREML: {
                "provider": "CoreML",
                "device": "Apple Neural Engine",
                "optimization": "FP16",
                "expected_latency_ms": "1-2"
            },
            AcceleratorType.NVIDIA_TENSORRT: {
                "provider": "TensorRT",
                "device": "NVIDIA GPU",
                "optimization": "INT8/FP16",
                "expected_latency_ms": "1-2"
            },
            AcceleratorType.NVIDIA_CUDA: {
                "provider": "CUDA",
                "device": "NVIDIA GPU",
                "optimization": "FP32",
                "expected_latency_ms": "2-3"
            },
            AcceleratorType.CPU_AVX512: {
                "provider": "ONNX Runtime",
                "device": "CPU (AVX-512)",
                "optimization": "INT8",
                "expected_latency_ms": "3-4"
            },
            AcceleratorType.CPU_ONNX: {
                "provider": "ONNX Runtime",
                "device": "CPU",
                "optimization": "FP32",
                "expected_latency_ms": "5-6"
            }
        }
        return info_map.get(accel_type, info_map[AcceleratorType.CPU_ONNX])


if __name__ == "__main__":
    # 测试硬件检测
    logging.basicConfig(level=logging.INFO)
    
    accel_type = HardwareDetector.detect()
    print(f"\n🔍 硬件检测结果：{accel_type.value}")
    
    packages = HardwareDetector.get_install_packages(accel_type)
    print(f"📦 需要安装的包：{', '.join(packages)}")
    
    info = HardwareDetector.get_accelerator_info(accel_type)
    print(f"🚀 加速方案：{info['provider']} on {info['device']}")
    print(f"⚡ 预期延迟：{info['expected_latency_ms']}ms/句子")
