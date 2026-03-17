# -*- coding: utf-8 -*-
import os
import time
import torch
from model import Model
from air_track.utils import combine_load_cfg_yaml
from collections import defaultdict


class LayerProfiler:
    def __init__(self, model):
        self.model = model
        self.handles = []
        self.timings = defaultdict(list)
        self.current_layer = None

    def _time_hook(self, name):
        def forward_pre_hook(module, input):
            torch.cuda.synchronize()
            self.current_layer = name
            self.timings[name].append({
                'start_time': time.time(),
                'start_memory': torch.cuda.memory_allocated()
            })

        def forward_hook(module, input, output):
            torch.cuda.synchronize()
            timing_data = self.timings[name][-1]
            timing_data['end_time'] = time.time()
            timing_data['end_memory'] = torch.cuda.memory_allocated()
            timing_data['duration'] = timing_data['end_time'] - timing_data['start_time']
            timing_data['memory_usage'] = (timing_data['end_memory'] - timing_data['start_memory']) / 1024 ** 2  # MB

            # 记录输出的显存使用
            if isinstance(output, torch.Tensor):
                timing_data['output_memory'] = output.element_size() * output.numel() / 1024 ** 2  # MB
            else:
                timing_data['output_memory'] = 0

        return forward_pre_hook, forward_hook

    def register_hooks(self):
        """为每个命名模块注册钩子"""
        for name, module in self.model.named_modules():
            if name:  # 跳过空名（根模块）
                pre_hook, hook = self._time_hook(name)
                handle_pre = module.register_forward_pre_hook(pre_hook)
                handle = module.register_forward_hook(hook)
                self.handles.extend([handle_pre, handle])

    def remove_hooks(self):
        """移除所有钩子"""
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def profile(self, input_tensor, num_runs=10, warmup=3):
        """执行性能分析"""
        self.model.eval()

        # Warmup
        print("Warming up...")
        with torch.no_grad():
            for _ in range(warmup):
                _ = self.model(input_tensor)

        # 注册钩子
        self.register_hooks()

        # 实际测试
        print(f"Profiling {num_runs} runs...")
        with torch.no_grad():
            for i in range(num_runs):
                torch.cuda.synchronize()
                _ = self.model(input_tensor)

        # 移除钩子
        self.remove_hooks()

        # 统计结果
        results = {}
        for layer_name, timings in self.timings.items():
            durations = [t['duration'] for t in timings]
            memory_usages = [t['memory_usage'] for t in timings]
            output_memories = [t.get('output_memory', 0) for t in timings]

            results[layer_name] = {
                'avg_time_ms': sum(durations) / len(durations) * 1000,
                'min_time_ms': min(durations) * 1000,
                'max_time_ms': max(durations) * 1000,
                'std_time_ms': (torch.std(torch.tensor(durations)) * 1000).item() if len(durations) > 1 else 0,
                'avg_memory_mb': sum(memory_usages) / len(memory_usages),
                'max_memory_mb': max(memory_usages),
                'avg_output_memory_mb': sum(output_memories) / len(output_memories),
                'num_calls': len(timings)
            }

        return results

    def print_summary(self, results, sort_by='avg_time_ms', top_k=20):
        """打印分析结果"""
        print("\n" + "=" * 120)
        print(
            f"{'Layer Name':<40} {'Avg Time (ms)':<15} {'Max Time (ms)':<15} {'Avg Memory (MB)':<15} {'Max Memory (MB)':<15} {'Output (MB)':<15}")
        print("-" * 120)

        sorted_results = sorted(results.items(), key=lambda x: x[1][sort_by], reverse=True)

        total_time = 0
        total_memory = 0
        total_output_memory = 0

        for layer_name, stats in sorted_results[:top_k]:
            print(f"{layer_name:<40} "
                  f"{stats['avg_time_ms']:<15.3f} "
                  f"{stats['max_time_ms']:<15.3f} "
                  f"{stats['avg_memory_mb']:<15.2f} "
                  f"{stats['max_memory_mb']:<15.2f} "
                  f"{stats['avg_output_memory_mb']:<15.2f}")

            total_time += stats['avg_time_ms']
            total_memory += stats['avg_memory_mb']
            total_output_memory += stats['avg_output_memory_mb']

        # 打印总计
        print("-" * 120)
        print(f"{'TOTAL (Sum)':<40} "
              f"{total_time:<15.3f} "
              f"{'':<15} "  # 跳过max time列
              f"{total_memory:<15.2f} "
              f"{'':<15} "  # 跳过max memory列
              f"{total_output_memory:<15.2f}")

        # 打印实际总显存使用（最大分配值）
        torch.cuda.synchronize()
        max_memory_allocated = torch.cuda.max_memory_allocated() / 1024 ** 2
        memory_allocated = torch.cuda.memory_allocated() / 1024 ** 2
        print("-" * 120)
        print(f"{'ACTUAL GPU MEMORY':<40} "
              f"{'':<15} {'':<15} "
              f"{memory_allocated:<15.2f} "
              f"{max_memory_allocated:<15.2f} "
              f"{'':<15}")
        print("=" * 120)

        return total_time, total_memory, total_output_memory

    def export_detailed_report(self, results, filename="layer_profiling_detailed.txt"):
        """导出详细报告"""
        with open(filename, "w") as f:
            f.write("=" * 120 + "\n")
            f.write("LAYER-WISE PROFILING DETAILED REPORT\n")
            f.write("=" * 120 + "\n\n")

            # 按耗时排序
            f.write("1. Sorted by Time Consumption:\n")
            f.write("-" * 120 + "\n")
            f.write(f"{'Layer Name':<40} {'Avg Time (ms)':<15} {'Memory (MB)':<15} {'Output (MB)':<15} {'Calls':<10}\n")
            f.write("-" * 120 + "\n")

            sorted_by_time = sorted(results.items(), key=lambda x: x[1]['avg_time_ms'], reverse=True)
            total_time = 0
            total_memory = 0

            for layer_name, stats in sorted_by_time:
                f.write(f"{layer_name:<40} "
                        f"{stats['avg_time_ms']:<15.3f} "
                        f"{stats['avg_memory_mb']:<15.2f} "
                        f"{stats['avg_output_memory_mb']:<15.2f} "
                        f"{stats['num_calls']:<10}\n")
                total_time += stats['avg_time_ms']
                total_memory += stats['avg_memory_mb']

            f.write("-" * 120 + "\n")
            f.write(f"{'TOTAL':<40} "
                    f"{total_time:<15.3f} "
                    f"{total_memory:<15.2f}\n")

            # 按显存使用排序
            f.write("\n\n2. Sorted by Memory Consumption:\n")
            f.write("-" * 120 + "\n")
            f.write(f"{'Layer Name':<40} {'Memory (MB)':<15} {'Time (ms)':<15} {'Output (MB)':<15}\n")
            f.write("-" * 120 + "\n")

            sorted_by_memory = sorted(results.items(), key=lambda x: x[1]['avg_memory_mb'], reverse=True)
            for layer_name, stats in sorted_by_memory[:30]:  # 只显示前30个
                f.write(f"{layer_name:<40} "
                        f"{stats['avg_memory_mb']:<15.2f} "
                        f"{stats['avg_time_ms']:<15.3f} "
                        f"{stats['avg_output_memory_mb']:<15.2f}\n")

            # 统计各模块类型
            f.write("\n\n3. Analysis by Module Type:\n")
            f.write("-" * 120 + "\n")

            module_stats = defaultdict(lambda: {'total_time': 0, 'total_memory': 0, 'count': 0})
            for layer_name, stats in results.items():
                # 根据层名推断类型
                parts = layer_name.split('.')
                if parts:
                    layer_type = parts[-1]
                    module_stats[layer_type]['total_time'] += stats['avg_time_ms']
                    module_stats[layer_type]['total_memory'] += stats['avg_memory_mb']
                    module_stats[layer_type]['count'] += 1

            f.write(
                f"{'Module Type':<20} {'Total Time (ms)':<15} {'Avg Time (ms)':<15} {'Total Memory (MB)':<15} {'Count':<10}\n")
            f.write("-" * 120 + "\n")

            for layer_type, data in sorted(module_stats.items(), key=lambda x: x[1]['total_time'], reverse=True):
                avg_time = data['total_time'] / data['count'] if data['count'] > 0 else 0
                f.write(f"{layer_type:<20} "
                        f"{data['total_time']:<15.2f} "
                        f"{avg_time:<15.2f} "
                        f"{data['total_memory']:<15.2f} "
                        f"{data['count']:<10}\n")

            # 添加GPU信息
            f.write("\n\n4. GPU Information:\n")
            f.write("-" * 120 + "\n")
            f.write(f"CUDA Available: {torch.cuda.is_available()}\n")
            if torch.cuda.is_available():
                f.write(f"GPU Device: {torch.cuda.get_device_name()}\n")
                f.write(f"Current Memory Allocated: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB\n")
                f.write(f"Max Memory Allocated: {torch.cuda.max_memory_allocated() / 1024 ** 2:.2f} MB\n")
                f.write(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB\n")

            f.write("\n" + "=" * 120 + "\n")
            f.write("End of Report\n")
            f.write("=" * 120 + "\n")

        print(f"\nDetailed report exported to: {filename}")


# 主程序
def main():
    # 获取配置
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_yaml = os.path.join(script_dir, 'config/template_detect_train.yaml')
    yaml_list = [train_yaml]
    cfg_data = combine_load_cfg_yaml(yaml_paths_list=yaml_list)

    # 创建模型
    model = Model(cfg_data['model_params']).cuda()

    # 准备输入
    input_tensor = torch.randn(2, 1, 512, 640).cuda()

    # 创建分析器
    profiler = LayerProfiler(model)

    # 执行分析
    print("Starting layer-wise profiling...")
    results = profiler.profile(input_tensor, num_runs=50, warmup=10)

    # 打印结果
    total_time, total_memory, total_output_memory = profiler.print_summary(results, top_k=30)

    # 导出详细报告
    profiler.export_detailed_report(results)

    # 打印简单的模块类型分析
    print("\n" + "=" * 120)
    print("Quick Analysis by Module Type:")
    print("=" * 120)

    module_stats = {}
    for layer_name, stats in results.items():
        # 根据层名推断类型
        parts = layer_name.split('.')
        if parts:
            layer_type = parts[-1]
            if layer_type not in module_stats:
                module_stats[layer_type] = {'total_time': 0, 'total_memory': 0, 'count': 0}
            module_stats[layer_type]['total_time'] += stats['avg_time_ms']
            module_stats[layer_type]['total_memory'] += stats['avg_memory_mb']
            module_stats[layer_type]['count'] += 1

    print(f"{'Module Type':<20} {'Total Time (ms)':<15} {'Avg Time (ms)':<15} {'Total Memory (MB)':<15} {'Count':<10}")
    print("-" * 120)

    for layer_type, data in sorted(module_stats.items(), key=lambda x: x[1]['total_time'], reverse=True):
        avg_time = data['total_time'] / data['count'] if data['count'] > 0 else 0
        avg_memory = data['total_memory'] / data['count'] if data['count'] > 0 else 0
        print(f"{layer_type:<20} "
              f"{data['total_time']:<15.2f} "
              f"{avg_time:<15.2f} "
              f"{data['total_memory']:<15.2f} "
              f"{data['count']:<10}")


if __name__ == "__main__":
    main()