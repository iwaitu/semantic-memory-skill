import grpc
from concurrent import futures
import os
import platform
import shutil
import time
import logging
import numpy as np
import onnxruntime as ort
import semantic_memory_pb2 as pb2
import semantic_memory_pb2_grpc as pb2_grpc
from transformers import AutoTokenizer
from hardware_detector import AcceleratorType, HardwareDetector

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
EMBEDDING_DIR_DEFAULT = os.path.join(MODELS_DIR, "qwen3-embedding-0.6b-onnx")
QWEN_RERANKER_DIR_DEFAULT = os.path.join(MODELS_DIR, "qwen3-reranker-seq-cls-onnx")


def _resolve_model_file(model_dir, filename="model.onnx"):
    return os.path.join(model_dir, filename)

# Model paths
EMBEDDING_MODEL_PATH = os.environ.get(
    "SEMANTIC_MEMORY_EMBEDDING_MODEL_PATH",
    _resolve_model_file(EMBEDDING_DIR_DEFAULT),
)
EMBEDDING_DIR = os.environ.get(
    "SEMANTIC_MEMORY_EMBEDDING_DIR",
    EMBEDDING_DIR_DEFAULT,
)
QWEN_RERANKER_MODEL_PATH = os.environ.get(
    "SEMANTIC_MEMORY_QWEN_RERANKER_MODEL_PATH",
    _resolve_model_file(QWEN_RERANKER_DIR_DEFAULT),
)
QWEN_RERANKER_DIR = os.environ.get(
    "SEMANTIC_MEMORY_QWEN_RERANKER_DIR",
    QWEN_RERANKER_DIR_DEFAULT,
)

QWEN_PREFIX = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
QWEN_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
DEFAULT_RERANK_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"

EMBEDDING_MAX_LENGTH = int(os.environ.get("SEMANTIC_MEMORY_EMBEDDING_MAX_LENGTH", "512"))
QWEN_RERANK_MAX_LENGTH = int(os.environ.get("SEMANTIC_MEMORY_QWEN_RERANK_MAX_LENGTH", "8192"))
COREML_CACHE_DIR = os.path.expanduser(
    os.environ.get(
        "SEMANTIC_MEMORY_COREML_CACHE_DIR",
        "~/Library/Caches/semantic-memory-skill/coreml",
    )
)
logger = logging.getLogger(__name__)

ACCELERATOR_OVERRIDE_MAP = {
    "coreml": AcceleratorType.APPLE_COREML,
    "cuda": AcceleratorType.NVIDIA_CUDA,
    "tensorrt": AcceleratorType.NVIDIA_TENSORRT,
    "cpu": AcceleratorType.CPU_ONNX,
    "cpu_avx512": AcceleratorType.CPU_AVX512,
}


def _env_flag(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _provider_chain_to_string(providers):
    labels = []
    for provider in providers:
        if isinstance(provider, tuple):
            provider_name, options = provider
            if provider_name == 'CoreMLExecutionProvider':
                labels.append(
                    f"CoreML(format={options.get('ModelFormat')},static={options.get('RequireStaticInputShapes')},units={options.get('MLComputeUnits')})"
                )
            elif provider_name == 'CUDAExecutionProvider':
                labels.append("CUDA")
            elif provider_name == 'TensorrtExecutionProvider':
                labels.append("TensorRT")
            else:
                labels.append(provider_name)
        else:
            labels.append(provider.replace('ExecutionProvider', ''))
    return " -> ".join(labels)


def _log_startup_banner(accelerator, worker_count):
    logger.info("=" * 72)
    logger.info(
        "Semantic Memory gRPC startup pid=%s accelerator=%s workers=%s",
        os.getpid(),
        accelerator.value,
        worker_count,
    )
    logger.info("Available ORT providers: %s", ort.get_available_providers())


def _log_model_session(model_name, providers, require_static_shapes):
    logger.info(
        "%s session: providers=%s static_shapes=%s",
        model_name,
        _provider_chain_to_string(providers),
        require_static_shapes,
    )


def _default_coreml_model_format():
    if platform.system() != "Darwin":
        return "NeuralNetwork"

    version = platform.mac_ver()[0]
    try:
        major = int(version.split(".")[0]) if version else 0
    except ValueError:
        major = 0
    return "MLProgram" if major >= 12 else "NeuralNetwork"


def _resolve_accelerator():
    override = os.environ.get("SEMANTIC_MEMORY_ACCELERATOR", "").strip().lower()
    if override:
        accelerator = ACCELERATOR_OVERRIDE_MAP.get(override)
        if accelerator is None:
            raise ValueError(f"Unsupported SEMANTIC_MEMORY_ACCELERATOR={override}")
        logger.info("Using accelerator override: %s", accelerator.value)
        return accelerator

    accelerator = HardwareDetector.detect()
    logger.info("Detected accelerator: %s", accelerator.value)
    return accelerator


def _get_coreml_provider_options(model_name, require_static_shapes):
    os.makedirs(COREML_CACHE_DIR, exist_ok=True)
    options = {
        "ModelFormat": os.environ.get(
            "SEMANTIC_MEMORY_COREML_MODEL_FORMAT",
            _default_coreml_model_format(),
        ),
        "MLComputeUnits": os.environ.get(
            "SEMANTIC_MEMORY_COREML_COMPUTE_UNITS",
            "ALL",
        ),
        "RequireStaticInputShapes": "1" if require_static_shapes else "0",
        "EnableOnSubgraphs": os.environ.get(
            "SEMANTIC_MEMORY_COREML_ENABLE_ON_SUBGRAPHS",
            "0",
        ),
        "ModelCacheDirectory": os.path.join(COREML_CACHE_DIR, model_name),
    }

    specialization_strategy = os.environ.get("SEMANTIC_MEMORY_COREML_SPECIALIZATION_STRATEGY")
    if specialization_strategy:
        options["SpecializationStrategy"] = specialization_strategy

    if _env_flag("SEMANTIC_MEMORY_COREML_PROFILE_COMPUTE_PLAN", default=False):
        options["ProfileComputePlan"] = "1"

    if _env_flag("SEMANTIC_MEMORY_COREML_ALLOW_LOW_PRECISION_GPU_ACCUMULATION", default=False):
        options["AllowLowPrecisionAccumulationOnGPU"] = "1"

    return options


def _create_session_options():
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session_options.log_severity_level = 3

    intra_threads = os.environ.get("SEMANTIC_MEMORY_ORT_INTRA_OP_THREADS")
    inter_threads = os.environ.get("SEMANTIC_MEMORY_ORT_INTER_OP_THREADS")
    if intra_threads:
        session_options.intra_op_num_threads = int(intra_threads)
    if inter_threads:
        session_options.inter_op_num_threads = int(inter_threads)

    return session_options


def _get_cuda_provider_options():
    options = {}
    if _env_flag("SEMANTIC_MEMORY_CUDA_USE_DEFAULT_STREAM", default=True):
        options["use_ep_level_unified_stream"] = "1"
    if _env_flag("SEMANTIC_MEMORY_CUDA_ENABLE_TUNING", default=False):
        options["tunable_op_enable"] = "1"
    return options


def _get_tensorrt_provider_options():
    options = {
        "trt_engine_cache_enable": "1",
        "trt_engine_cache_path": os.path.expanduser(
            os.environ.get(
                "SEMANTIC_MEMORY_TENSORRT_CACHE_DIR",
                "~/.cache/semantic-memory-skill/tensorrt",
            )
        ),
    }
    if _env_flag("SEMANTIC_MEMORY_TENSORRT_FP16", default=True):
        options["trt_fp16_enable"] = "1"
    return options


def get_optimal_providers(
    accelerator,
    model_name,
    require_static_shapes=False,
    model_format=None,
):
    available = ort.get_available_providers()
    if accelerator == AcceleratorType.APPLE_COREML and 'CoreMLExecutionProvider' in available:
        coreml_options = _get_coreml_provider_options(model_name, require_static_shapes)
        if model_format:
            coreml_options["ModelFormat"] = model_format
        return [
            ('CoreMLExecutionProvider', coreml_options),
            'CPUExecutionProvider',
        ]

    if accelerator == AcceleratorType.NVIDIA_TENSORRT and 'TensorrtExecutionProvider' in available:
        return [
            ('TensorrtExecutionProvider', _get_tensorrt_provider_options()),
            ('CUDAExecutionProvider', _get_cuda_provider_options()),
            'CPUExecutionProvider',
        ]

    if accelerator in {AcceleratorType.NVIDIA_TENSORRT, AcceleratorType.NVIDIA_CUDA} and 'CUDAExecutionProvider' in available:
        return [
            ('CUDAExecutionProvider', _get_cuda_provider_options()),
            'CPUExecutionProvider',
        ]

    return ['CPUExecutionProvider']


def _clear_coreml_cache(model_name):
    cache_path = os.path.join(COREML_CACHE_DIR, model_name)
    if os.path.isdir(cache_path):
        shutil.rmtree(cache_path, ignore_errors=True)


def _get_provider_candidates(accelerator, model_name, require_static_shapes=False):
    available = ort.get_available_providers()
    if accelerator == AcceleratorType.APPLE_COREML and 'CoreMLExecutionProvider' in available:
        preferred_format = os.environ.get(
            "SEMANTIC_MEMORY_COREML_MODEL_FORMAT",
            _default_coreml_model_format(),
        )
        candidates = [
            get_optimal_providers(
                accelerator,
                model_name,
                require_static_shapes=require_static_shapes,
                model_format=preferred_format,
            )
        ]
        if preferred_format == "MLProgram":
            candidates.append(
                get_optimal_providers(
                    accelerator,
                    model_name,
                    require_static_shapes=require_static_shapes,
                    model_format="NeuralNetwork",
                )
            )
        candidates.append(['CPUExecutionProvider'])
        return candidates

    if accelerator == AcceleratorType.NVIDIA_TENSORRT and 'TensorrtExecutionProvider' in available:
        return [
            get_optimal_providers(accelerator, model_name, require_static_shapes=require_static_shapes),
            [
                ('CUDAExecutionProvider', _get_cuda_provider_options()),
                'CPUExecutionProvider',
            ],
            ['CPUExecutionProvider'],
        ]

    if accelerator in {AcceleratorType.NVIDIA_TENSORRT, AcceleratorType.NVIDIA_CUDA} and 'CUDAExecutionProvider' in available:
        return [
            get_optimal_providers(accelerator, model_name, require_static_shapes=require_static_shapes),
            ['CPUExecutionProvider'],
        ]

    if accelerator in {AcceleratorType.CPU_AVX512, AcceleratorType.CPU_ONNX}:
        return [['CPUExecutionProvider']]

    return [['CPUExecutionProvider']]


def create_inference_session(accelerator, model_path, model_name, require_static_shapes=False):
    last_error = None
    for providers in _get_provider_candidates(
        accelerator,
        model_name,
        require_static_shapes=require_static_shapes,
    ):
        try:
            session = ort.InferenceSession(
                model_path,
                sess_options=_create_session_options(),
                providers=providers,
            )
            return session, providers
        except Exception as exc:
            last_error = exc
            if providers and isinstance(providers[0], tuple) and providers[0][0] == 'CoreMLExecutionProvider':
                logger.warning(
                    "Failed to initialize %s with CoreML providers %s: %s",
                    model_name,
                    providers,
                    exc,
                )
                _clear_coreml_cache(model_name)
                continue
            if providers and isinstance(providers[0], tuple) and providers[0][0] == 'TensorrtExecutionProvider':
                logger.warning(
                    "Failed to initialize %s with TensorRT providers %s: %s",
                    model_name,
                    providers,
                    exc,
                )
                continue
            raise

    raise last_error


def get_default_worker_count(accelerator):
    max_workers = os.environ.get("SEMANTIC_MEMORY_GRPC_MAX_WORKERS")
    if max_workers:
        return int(max_workers)
    if accelerator == AcceleratorType.APPLE_COREML:
        return min(4, os.cpu_count() or 4)
    if accelerator in {AcceleratorType.NVIDIA_TENSORRT, AcceleratorType.NVIDIA_CUDA}:
        return min(8, os.cpu_count() or 8)
    return 10

class SemanticMemoryService(pb2_grpc.SemanticMemoryServicer):
    def __init__(self, accelerator):
        self.accelerator = accelerator
        # Initialize tokenizers
        self.tokenizer_emb = AutoTokenizer.from_pretrained(EMBEDDING_DIR)
        self.tokenizer_qwen_rerank = AutoTokenizer.from_pretrained(
            QWEN_RERANKER_DIR,
            padding_side="left",
            trust_remote_code=True,
        )

        self.embedding_static_shapes = _env_flag(
            "SEMANTIC_MEMORY_COREML_STATIC_EMBEDDING",
            default=self.accelerator == AcceleratorType.APPLE_COREML,
        )
        self.qwen_static_shapes = _env_flag(
            "SEMANTIC_MEMORY_COREML_STATIC_QWEN_RERANK",
            default=False,
        )
        
        # Initialize ONNX sessions
        self.embedding_session, embedding_providers = create_inference_session(
            self.accelerator,
            EMBEDDING_MODEL_PATH,
            "embedding",
            require_static_shapes=self.embedding_static_shapes,
        )
        self.qwen_reranker_session, qwen_providers = create_inference_session(
            self.accelerator,
            QWEN_RERANKER_MODEL_PATH,
            "qwen-reranker",
            require_static_shapes=self.qwen_static_shapes,
        )
        accelerator_info = HardwareDetector.get_accelerator_info(self.accelerator)
        logger.info("Accelerator info: %s", accelerator_info)
        _log_model_session("Embedding", embedding_providers, self.embedding_static_shapes)
        _log_model_session("Qwen reranker", qwen_providers, self.qwen_static_shapes)
        logger.info("Models loaded successfully.")

    def _get_padding_mode(self, use_static_shapes):
        return "max_length" if use_static_shapes else True

    def Embedding(self, request, context):
        try:
            texts = list(request.texts)
            # 输入长度校验
            max_len = EMBEDDING_MAX_LENGTH
            for i, text in enumerate(texts):
                if len(text) > max_len * 4:  # 粗略估计
                    context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                    context.set_details(f"Text {i} exceeds max length")
                    return pb2.EmbeddingResponse()

            inputs = self.tokenizer_emb(
                texts,
                padding=self._get_padding_mode(self.embedding_static_shapes),
                truncation=True,
                max_length=EMBEDDING_MAX_LENGTH,
                return_tensors="np",
            )
            # Get input names from model
            input_names = [i.name for i in self.embedding_session.get_inputs()]
            input_feed = {k: inputs[k].astype(np.int64) for k in input_names if k in inputs}
            
            # 推理
            outputs = self.embedding_session.run(None, input_feed)
            embeddings = outputs[0]
            
            # ✅ 显式清理输入对象
            del inputs
            del input_feed
            
            if embeddings.ndim == 3:
                embeddings = embeddings[:, 0, :]
                
            if request.normalize:
                norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
                embeddings = embeddings / (norm + 1e-9)
                
            return pb2.EmbeddingResponse(embeddings=embeddings.flatten().tolist(), dimension=embeddings.shape[-1])
        except Exception as e:
            # 异常时清理
            try:
                del inputs
                del input_feed
            except:
                pass
            raise e

    def _validate_rerank_request(self, request, context):
        max_len = min(QWEN_RERANK_MAX_LENGTH, 2048)
        if len(request.query) > max_len * 4:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("Query exceeds max length")
            return False
        for i, doc in enumerate(request.documents):
            if len(doc) > max_len * 4:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details(f"Document {i} exceeds max length")
                return False
        return True

    def _format_qwen_instruction(self, instruction, query, document):
        rerank_instruction = instruction or DEFAULT_RERANK_INSTRUCTION
        return (
            f"{QWEN_PREFIX}<Instruct>: {rerank_instruction}\n"
            f"<Query>: {query}\n"
            f"<Document>: {document}{QWEN_SUFFIX}"
        )

    def _build_rerank_response(self, request, scores):
        results = []
        for i, (doc, score) in enumerate(zip(request.documents, scores)):
            results.append(pb2.RankedDocument(index=i, text=doc, score=float(score)))

        results.sort(key=lambda item: item.score, reverse=True)
        if request.top_k > 0:
            results = results[:request.top_k]
        return pb2.RerankResponse(results=results)

    def _run_qwen_rerank(self, request):
        pairs = [
            self._format_qwen_instruction(request.instruction, request.query, doc)
            for doc in request.documents
        ]
        encodings = self.tokenizer_qwen_rerank(
            pairs,
            padding=self._get_padding_mode(self.qwen_static_shapes),
            truncation=True,
            max_length=QWEN_RERANK_MAX_LENGTH,
            return_tensors="np",
        )
        input_names = {item.name for item in self.qwen_reranker_session.get_inputs()}
        input_feed = {
            key: value.astype(np.int64)
            for key, value in encodings.items()
            if key in input_names
        }
        logits = np.asarray(self.qwen_reranker_session.run(None, input_feed)[0]).squeeze(-1)
        scores = 1.0 / (1.0 + np.exp(-logits))
        return np.asarray(scores).reshape(-1), encodings, input_feed

    def Rerank(self, request, context):
        return self.RerankQwen(request, context)

    def RerankQwen(self, request, context):
        try:
            if not self._validate_rerank_request(request, context):
                return pb2.RerankResponse()
            scores, encodings, input_feed = self._run_qwen_rerank(request)

            del encodings
            del input_feed

            return self._build_rerank_response(request, scores)
        except Exception as e:
            try:
                del encodings
                del input_feed
            except:
                pass
            context.set_details(str(e))
            context.set_code(grpc.StatusCode.INTERNAL)
            return pb2.RerankResponse()

    def RetrieveAndRerank(self, request, context):
        try:
            start_time = time.time()
            
            # 1. Embedding + Cosine Similarity (Assume we have stored embeddings in-memory for this skill)
            # For this task, we will simulate embedding retrieval
            emb_req = pb2.EmbeddingRequest(texts=[request.query], normalize=True)
            query_emb = self.Embedding(emb_req, context).embeddings
            
            # 2. Rerank
            rerank_req = pb2.RerankRequest(
                query=request.query,
                documents=request.documents,
                top_k=request.top_k,
            )
            rerank_res = self.RerankQwen(rerank_req, context)
            
            total_latency = (time.time() - start_time) * 1000
            return pb2.RetrieveAndRerankResponse(results=rerank_res.results, total_latency_ms=total_latency)
        except Exception as e:
            context.set_details(str(e))
            context.set_code(grpc.StatusCode.INTERNAL)
            return pb2.RetrieveAndRerankResponse()

    def Health(self, request, context):
        return pb2.HealthResponse(healthy=True, version="1.0.0")

def serve():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s:%(name)s:%(message)s",
    )
    accelerator = _resolve_accelerator()
    worker_count = get_default_worker_count(accelerator)
    _log_startup_banner(accelerator, worker_count)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=worker_count))
    pb2_grpc.add_SemanticMemoryServicer_to_server(SemanticMemoryService(accelerator), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    logger.info("gRPC server started on port 50051.")
    server.wait_for_termination()

if __name__ == '__main__':
    serve()
