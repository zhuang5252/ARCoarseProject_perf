"""
简单的 ONNX -> TensorRT 基准脚本

用法示例:
  python tools/benchmark_tensorrt.py --onnx model.onnx --batch 1 --precision fp16 --iters 200

依赖: tensorrt, pycuda, numpy
"""
import argparse
import time
import numpy as np

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit  # noqa: F401
except Exception as e:
    raise RuntimeError('requires tensorrt and pycuda: %s' % e)

TRT_LOGGER = trt.Logger(trt.Logger.INFO)


def build_engine_from_onnx(onnx_path: str, precision: str = 'fp32'):
    builder = trt.Builder(TRT_LOGGER)
    network_flags = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            errs = []
            for i in range(parser.num_errors):
                errs.append(str(parser.get_error(i)))
            raise RuntimeError('ONNX parse failed:\n' + '\n'.join(errs))

    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 30  # 1GB
    if precision == 'fp16':
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
    elif precision == 'int8':
        # INT8 requires calibration; user must extend for calibration
        if builder.platform_has_fast_int8:
            config.set_flag(trt.BuilderFlag.INT8)
    profile = builder.create_optimization_profile()

    # set profile from network input shapes (take first input)
    inp = network.get_input(0)
    shape = inp.shape  # may contain -1
    if any([d == -1 for d in shape]):
        # set a reasonable profile -- user can customize
        min_shape = tuple(1 if d == -1 else d for d in shape)
        opt_shape = tuple(1 if d == -1 else d for d in shape)
        max_shape = tuple(8 if d == -1 else d for d in shape)
        profile.set_shape(inp.name, min_shape, opt_shape, max_shape)
        config.add_optimization_profile(profile)

    engine = builder.build_engine(network, config)
    if engine is None:
        raise RuntimeError('Failed to build engine')
    return engine


def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()

    for i in range(engine.num_bindings):
        name = engine.get_binding_name(i)
        size = trt.volume(engine.get_binding_shape(i)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(i))
        # allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        dev_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(dev_mem))
        if engine.binding_is_input(i):
            inputs.append({'name': name, 'host': host_mem, 'device': dev_mem, 'dtype': dtype})
        else:
            outputs.append({'name': name, 'host': host_mem, 'device': dev_mem, 'dtype': dtype})

    return inputs, outputs, bindings, stream


def infer_context(engine, inputs, outputs, bindings, stream, context, iters=200, warmup=20):
    # prepare random input in host buffers
    for inp in inputs:
        arr = np.random.random_sample(inp['host'].shape).astype(inp['dtype'])
        np.copyto(inp['host'], arr)

    # copy input to device
    for inp in inputs:
        cuda.memcpy_htod_async(inp['device'], inp['host'], stream)

    # warmup
    for _ in range(warmup):
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    cuda.Context.synchronize()

    # timed runs
    t0 = time.perf_counter()
    for _ in range(iters):
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    cuda.Context.synchronize()
    t1 = time.perf_counter()

    avg_ms = (t1 - t0) * 1000.0 / iters
    return avg_ms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx', required=True, help='input ONNX model')
    parser.add_argument('--precision', choices=['fp32', 'fp16', 'int8'], default='fp32')
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--iters', type=int, default=200)
    parser.add_argument('--warmup', type=int, default=20)
    args = parser.parse_args()

    print('Building engine (this may take some time)...')
    engine = build_engine_from_onnx(args.onnx, precision=args.precision)
    print('Engine built. Creating execution context...')
    context = engine.create_execution_context()

    inputs, outputs, bindings, stream = allocate_buffers(engine)

    avg_ms = infer_context(engine, inputs, outputs, bindings, stream, context, iters=args.iters, warmup=args.warmup)
    print('Average latency: %.3f ms (batch=%d, precision=%s)' % (avg_ms, args.batch, args.precision))


if __name__ == '__main__':
    main()
