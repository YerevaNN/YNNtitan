# AOT ID: ['0_forward']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
from torch._inductor.codegen.multi_kernel import MultiKernelCall
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid, split_scan_grid, grid_combo_kernels, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()


# kernel path: /auto/home/menuab/code/YNNtitan/torchinductor_menuab/t3/ct3ummmx75d5nnzayexvc3s42exqjbbuomtqszsbepkdnn5sqk67.py
# Topologically Sorted Source Nodes: [float_1, pow_1, mean, add, rsqrt, mul, output, mul_1], Original ATen: [aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul]
# Source node to ATen node mapping:
#   add => add
#   float_1 => convert_element_type
#   mean => mean
#   mul => mul
#   mul_1 => mul_1
#   output => convert_element_type_1
#   pow_1 => pow_1
#   rsqrt => rsqrt
# Graph fragment:
#   %convert_element_type : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_2, torch.float32), kwargs = {})
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type, 2), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_1, [-1], True), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean, 1e-05), kwargs = {})
#   %rsqrt : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add,), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type, %rsqrt), kwargs = {})
#   %convert_element_type_1 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul, torch.bfloat16), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_1, %primals_1), kwargs = {})
triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.reduction(
    size_hints=[8192, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: 'i32', 5: 'i32'}, 'device': DeviceProperties(type='cuda', index=1, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_0', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 3, 'num_reduction': 1, 'backend_hash': '54E46422D5DB2E55B804C8E038A4A0E2ECEED6FCC5402DED453936C14F5DFA13', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (2048*x0)), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = tmp1 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tmp6 = 2048.0
    tmp7 = tmp4 / tmp6
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = libdevice.rsqrt(tmp9)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp10, None)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp11 = tl.load(in_ptr0 + (r1 + (2048*x0)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp15 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp12 = tmp11.to(tl.float32)
        tmp13 = tmp12 * tmp10
        tmp14 = tmp13.to(tl.float32)
        tmp16 = tmp14 * tmp15
        tl.store(out_ptr0 + (r1 + (2048*x0)), tmp16, rmask)
''', device_str='cuda')


# kernel path: /auto/home/menuab/code/YNNtitan/torchinductor_menuab/xd/cxdrlih7n4xmq7tve7fwzouoqlkbvzebsxsovbdp5m7t2bpeznde.py
# Topologically Sorted Source Nodes: [xq_], Original ATen: [aten.view_as_complex]
# Source node to ATen node mapping:
#   xq_ => view_as_complex
# Graph fragment:
#   %view_as_complex : [num_users=1] = call_function[target=torch.ops.aten.view_as_complex.default](args = (%view_9,), kwargs = {})
triton_poi_fused_view_as_complex_1 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[16777216], 
    filename=__file__,
    triton_meta={'signature': {0: '*bf16', 1: '*fp32', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=1, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_view_as_complex_1', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '54E46422D5DB2E55B804C8E038A4A0E2ECEED6FCC5402DED453936C14F5DFA13', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16777216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /auto/home/menuab/code/YNNtitan/torchinductor_menuab/yq/cyqr6kkbznmqn3hfr67ymzmaxvzgfazpou3r4aeuprzh2wnm6qzv.py
# Topologically Sorted Source Nodes: [xk_], Original ATen: [aten.view_as_complex]
# Source node to ATen node mapping:
#   xk_ => view_as_complex_1
# Graph fragment:
#   %view_as_complex_1 : [num_users=1] = call_function[target=torch.ops.aten.view_as_complex.default](args = (%view_10,), kwargs = {})
triton_poi_fused_view_as_complex_2 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[4194304], 
    filename=__file__,
    triton_meta={'signature': {0: '*bf16', 1: '*fp32', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=1, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_view_as_complex_2', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '54E46422D5DB2E55B804C8E038A4A0E2ECEED6FCC5402DED453936C14F5DFA13', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /auto/home/menuab/code/YNNtitan/torchinductor_menuab/ia/ciaedtacmwksbkchmag2ftegtuoknfojp6id5eagmg7shxy2xk4l.py
# Topologically Sorted Source Nodes: [xq_2], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   xq_2 => convert_element_type_10
# Graph fragment:
#   %convert_element_type_10 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_12, torch.bfloat16), kwargs = {})
triton_poi_fused__to_copy_3 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[16777216], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*bf16', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=1, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_3', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '54E46422D5DB2E55B804C8E038A4A0E2ECEED6FCC5402DED453936C14F5DFA13', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16777216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /auto/home/menuab/code/YNNtitan/torchinductor_menuab/yg/cygmjhv6levlxvodypl23qm4df66qmcbeln522iyajm7nziykr2w.py
# Topologically Sorted Source Nodes: [keys], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   keys => clone
# Graph fragment:
#   %clone : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_4 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[16777216], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*bf16', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=1, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_4', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '54E46422D5DB2E55B804C8E038A4A0E2ECEED6FCC5402DED453936C14F5DFA13', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16777216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 64
    x2 = (xindex // 256)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*x2)), None, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x3), tmp1, None)
''', device_str='cuda')


# kernel path: /auto/home/menuab/code/YNNtitan/torchinductor_menuab/hp/chp5jrr4dptyf2rtzspmprdfc3bzbyagq5kcxockxucqokwt3mpv.py
# Topologically Sorted Source Nodes: [values], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   values => clone_1
# Graph fragment:
#   %clone_1 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_1,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_5 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[16777216], 
    filename=__file__,
    triton_meta={'signature': {0: '*bf16', 1: '*bf16', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=1, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_5', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '54E46422D5DB2E55B804C8E038A4A0E2ECEED6FCC5402DED453936C14F5DFA13', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16777216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 64
    x2 = (xindex // 256)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*x2)), None, eviction_policy='evict_last').to(tl.float32)
    tl.store(out_ptr0 + (x3), tmp0, None)
''', device_str='cuda')


# kernel path: /auto/home/menuab/code/YNNtitan/torchinductor_menuab/2h/c2hniu52unno2wwefimepwu3duq65mlw3432b6aj35erqyulhraf.py
# Topologically Sorted Source Nodes: [h, float_4, pow_2, mean_1, add_2, rsqrt_1, mul_4, output_4, mul_5], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
# Source node to ATen node mapping:
#   add_2 => add_2
#   float_4 => convert_element_type_14
#   h => add_1
#   mean_1 => mean_1
#   mul_4 => mul_4
#   mul_5 => mul_5
#   output_4 => convert_element_type_15
#   pow_2 => pow_2
#   rsqrt_1 => rsqrt_1
# Graph fragment:
#   %add_1 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_2, %view_18), kwargs = {})
#   %convert_element_type_14 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_1, torch.float32), kwargs = {})
#   %pow_2 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_14, 2), kwargs = {})
#   %mean_1 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_2, [-1], True), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_1, 1e-05), kwargs = {})
#   %rsqrt_1 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_2,), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_14, %rsqrt_1), kwargs = {})
#   %convert_element_type_15 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_4, torch.bfloat16), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_15, %primals_8), kwargs = {})
triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_6 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.reduction(
    size_hints=[8192, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: '*bf16', 5: 'i32', 6: 'i32'}, 'device': DeviceProperties(type='cuda', index=1, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_6', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 5, 'num_reduction': 1, 'backend_hash': '54E46422D5DB2E55B804C8E038A4A0E2ECEED6FCC5402DED453936C14F5DFA13', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (2048*x0)), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r1 + (2048*x0)), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp3 = tmp2.to(tl.float32)
        tmp4 = tmp3 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tmp8 = 2048.0
    tmp9 = tmp6 / tmp8
    tmp10 = 1e-05
    tmp11 = tmp9 + tmp10
    tmp12 = libdevice.rsqrt(tmp11)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp12, None)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp13 = tl.load(in_ptr0 + (r1 + (2048*x0)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp14 = tl.load(in_ptr1 + (r1 + (2048*x0)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp19 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp15 = tmp13 + tmp14
        tmp16 = tmp15.to(tl.float32)
        tmp17 = tmp16 * tmp12
        tmp18 = tmp17.to(tl.float32)
        tmp20 = tmp18 * tmp19
        tl.store(out_ptr0 + (r1 + (2048*x0)), tmp20, rmask)
''', device_str='cuda')


# kernel path: /auto/home/menuab/code/YNNtitan/torchinductor_menuab/y3/cy36k2vg46emfs7wouylif3esx3sspeh737ssupfmmjaw2ef4zdf.py
# Topologically Sorted Source Nodes: [silu, mul_6], Original ATen: [aten.silu, aten.mul]
# Source node to ATen node mapping:
#   mul_6 => mul_7
#   silu => convert_element_type_18, convert_element_type_19, mul_6, sigmoid
# Graph fragment:
#   %convert_element_type_18 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_20, torch.float32), kwargs = {})
#   %sigmoid : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_18,), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_18, %sigmoid), kwargs = {})
#   %convert_element_type_19 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_6, torch.bfloat16), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_19, %view_22), kwargs = {})
triton_poi_fused_mul_silu_7 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[67108864], 
    filename=__file__,
    triton_meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=1, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_silu_7', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '54E46422D5DB2E55B804C8E038A4A0E2ECEED6FCC5402DED453936C14F5DFA13', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 67108864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp5 = tl.load(in_ptr1 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp1 * tmp2
    tmp4 = tmp3.to(tl.float32)
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr0 + (x0), tmp6, None)
''', device_str='cuda')


# kernel path: /auto/home/menuab/code/YNNtitan/torchinductor_menuab/fc/cfcsyrya5foxqlusqjdamtzvu6mzfc4wtzbbkmbwdeqc7j2434f4.py
# Topologically Sorted Source Nodes: [h, out], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   h => add_1
#   out => add_3
# Graph fragment:
#   %add_1 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_2, %view_18), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_1, %view_24), kwargs = {})
triton_poi_fused_add_8 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[16777216], 
    filename=__file__,
    triton_meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=1, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_8', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '54E46422D5DB2E55B804C8E038A4A0E2ECEED6FCC5402DED453936C14F5DFA13', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16777216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x0), None).to(tl.float32)
    tmp3 = tl.load(in_out_ptr0 + (x0), None).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tl.store(in_out_ptr0 + (x0), tmp4, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11 = args
    args.clear()
    assert_size_stride(primals_1, (2048, ), (1, ))
    assert_size_stride(primals_2, (4, 2048, 2048), (4194304, 2048, 1))
    assert_size_stride(primals_3, (2048, 2048), (2048, 1))
    assert_size_stride(primals_4, (512, 2048), (2048, 1))
    assert_size_stride(primals_5, (512, 2048), (2048, 1))
    assert_size_stride(primals_6, (4096, 32), (32, 1))
    assert_size_stride(primals_7, (2048, 2048), (2048, 1))
    assert_size_stride(primals_8, (2048, ), (1, ))
    assert_size_stride(primals_9, (8192, 2048), (2048, 1))
    assert_size_stride(primals_10, (8192, 2048), (2048, 1))
    assert_size_stride(primals_11, (2048, 8192), (8192, 1))
    with torch.cuda._DeviceGuard(1):
        torch.cuda.set_device(1)
        buf0 = empty_strided_cuda((4, 2048, 1), (2048, 1, 8192), torch.float32)
        buf1 = reinterpret_tensor(buf0, (4, 2048, 1), (2048, 1, 1), 0); del buf0  # reuse
        buf2 = empty_strided_cuda((4, 2048, 2048), (4194304, 2048, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [float_1, pow_1, mean, add, rsqrt, mul, output, mul_1], Original ATen: [aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul]
        stream1 = get_raw_stream(1)
        triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_0.run(buf1, primals_2, primals_1, buf2, 8192, 2048, grid=grid(8192), stream=stream1)
        buf3 = empty_strided_cuda((8192, 2048), (2048, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [xq], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf2, (8192, 2048), (2048, 1), 0), reinterpret_tensor(primals_3, (2048, 2048), (1, 2048), 0), out=buf3)
        buf4 = empty_strided_cuda((8192, 512), (512, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [xk], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf2, (8192, 2048), (2048, 1), 0), reinterpret_tensor(primals_4, (2048, 512), (1, 2048), 0), out=buf4)
        buf5 = empty_strided_cuda((8192, 512), (512, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [xv], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf2, (8192, 2048), (2048, 1), 0), reinterpret_tensor(primals_5, (2048, 512), (1, 2048), 0), out=buf5)
        buf6 = empty_strided_cuda((4, 2048, 32, 32, 2), (4194304, 2048, 64, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [xq_], Original ATen: [aten.view_as_complex]
        triton_poi_fused_view_as_complex_1.run(buf3, buf6, 16777216, grid=grid(16777216), stream=stream1)
        # Topologically Sorted Source Nodes: [xq_], Original ATen: [aten.view_as_complex]
        buf7 = torch.ops.aten.view_as_complex.default(buf6)
        buf8 = buf7
        buf9 = empty_strided_cuda((4, 2048, 8, 32, 2), (1048576, 512, 64, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [xk_], Original ATen: [aten.view_as_complex]
        triton_poi_fused_view_as_complex_2.run(buf4, buf9, 4194304, grid=grid(4194304), stream=stream1)
        del buf4
        # Topologically Sorted Source Nodes: [xk_], Original ATen: [aten.view_as_complex]
        buf10 = torch.ops.aten.view_as_complex.default(buf9)
        buf11 = buf10
        # Topologically Sorted Source Nodes: [freqs_cis], Original ATen: [aten.slice]
        buf12 = torch.ops.aten.slice.Tensor(primals_6, 0, 0, 2048)
        buf13 = buf12
        # Topologically Sorted Source Nodes: [freqs_cis_1], Original ATen: [aten.view]
        buf14 = torch.ops.aten.reshape.default(buf13, [1, 2048, 1, 32])
        buf15 = buf14
        # Topologically Sorted Source Nodes: [mul_2], Original ATen: [aten.mul]
        buf16 = torch.ops.aten.mul.Tensor(buf8, buf15)
        del buf6
        del buf7
        del buf8
        buf17 = buf16
        del buf16
        # Topologically Sorted Source Nodes: [view_as_real], Original ATen: [aten.view_as_real]
        buf18 = torch.ops.aten.view_as_real.default(buf17)
        buf19 = buf18
        # Topologically Sorted Source Nodes: [mul_3], Original ATen: [aten.mul]
        buf20 = torch.ops.aten.mul.Tensor(buf11, buf15)
        del buf10
        del buf11
        del buf9
        buf21 = buf20
        del buf20
        # Topologically Sorted Source Nodes: [view_as_real_1], Original ATen: [aten.view_as_real]
        buf22 = torch.ops.aten.view_as_real.default(buf21)
        buf23 = buf22
        buf24 = reinterpret_tensor(buf3, (4, 2048, 32, 64), (4194304, 2048, 64, 1), 0); del buf3  # reuse
        # Topologically Sorted Source Nodes: [xq_2], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_3.run(buf19, buf24, 16777216, grid=grid(16777216), stream=stream1)
        del buf17
        del buf18
        del buf19
        buf25 = empty_strided_cuda((4, 2048, 8, 4, 64), (4194304, 2048, 256, 64, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [keys], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf23, buf25, 16777216, grid=grid(16777216), stream=stream1)
        del buf21
        del buf22
        del buf23
        buf26 = empty_strided_cuda((4, 2048, 8, 4, 64), (4194304, 2048, 256, 64, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [values], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf5, buf26, 16777216, grid=grid(16777216), stream=stream1)
        del buf5
        # Topologically Sorted Source Nodes: [output_1], Original ATen: [aten._scaled_dot_product_flash_attention]
        buf27 = torch.ops.aten._scaled_dot_product_flash_attention.default(reinterpret_tensor(buf24, (4, 32, 2048, 64), (4194304, 64, 2048, 1), 0), reinterpret_tensor(buf25, (4, 32, 2048, 64), (4194304, 64, 2048, 1), 0), reinterpret_tensor(buf26, (4, 32, 2048, 64), (4194304, 64, 2048, 1), 0), 0.0, True, scale=0.125)
        buf28 = buf27[0]
        buf29 = buf27[1]
        buf30 = buf27[6]
        buf31 = buf27[7]
        del buf27
        buf33 = empty_strided_cuda((8192, 2048), (2048, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf28, (8192, 2048), (2048, 1), 0), reinterpret_tensor(primals_7, (2048, 2048), (1, 2048), 0), out=buf33)
        buf34 = empty_strided_cuda((4, 2048, 1), (2048, 1, 8192), torch.float32)
        buf35 = reinterpret_tensor(buf34, (4, 2048, 1), (2048, 1, 1), 0); del buf34  # reuse
        buf36 = empty_strided_cuda((4, 2048, 2048), (4194304, 2048, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [h, float_4, pow_2, mean_1, add_2, rsqrt_1, mul_4, output_4, mul_5], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_6.run(buf35, primals_2, buf33, primals_8, buf36, 8192, 2048, grid=grid(8192), stream=stream1)
        buf37 = empty_strided_cuda((8192, 8192), (8192, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_4], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf36, (8192, 2048), (2048, 1), 0), reinterpret_tensor(primals_9, (2048, 8192), (1, 2048), 0), out=buf37)
        buf38 = empty_strided_cuda((8192, 8192), (8192, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_5], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf36, (8192, 2048), (2048, 1), 0), reinterpret_tensor(primals_10, (2048, 8192), (1, 2048), 0), out=buf38)
        buf39 = empty_strided_cuda((4, 2048, 8192), (16777216, 8192, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [silu, mul_6], Original ATen: [aten.silu, aten.mul]
        triton_poi_fused_mul_silu_7.run(buf37, buf38, buf39, 67108864, grid=grid(67108864), stream=stream1)
        buf40 = empty_strided_cuda((8192, 2048), (2048, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_6], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf39, (8192, 8192), (8192, 1), 0), reinterpret_tensor(primals_11, (8192, 2048), (1, 8192), 0), out=buf40)
        buf41 = reinterpret_tensor(buf40, (4, 2048, 2048), (4194304, 2048, 1), 0); del buf40  # reuse
        # Topologically Sorted Source Nodes: [h, out], Original ATen: [aten.add]
        triton_poi_fused_add_8.run(buf41, primals_2, buf33, 16777216, grid=grid(16777216), stream=stream1)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._conj]
        buf42 = torch.ops.aten._conj.default(buf15)
        buf43 = buf42
        del primals_6
    return (buf41, primals_1, primals_2, primals_8, buf1, reinterpret_tensor(buf2, (8192, 2048), (2048, 1), 0), reinterpret_tensor(buf24, (4, 32, 2048, 64), (4194304, 64, 2048, 1), 0), reinterpret_tensor(buf25, (4, 32, 2048, 64), (4194304, 64, 2048, 1), 0), reinterpret_tensor(buf26, (4, 32, 2048, 64), (4194304, 64, 2048, 1), 0), buf28, buf29, buf30, buf31, buf33, buf35, reinterpret_tensor(buf36, (8192, 2048), (2048, 1), 0), buf37, buf38, reinterpret_tensor(buf39, (8192, 8192), (8192, 1), 0), primals_11, primals_10, primals_9, primals_7, buf43, primals_5, primals_4, primals_3, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((2048, ), (1, ), device='cuda:1', dtype=torch.bfloat16)
    primals_2 = rand_strided((4, 2048, 2048), (4194304, 2048, 1), device='cuda:1', dtype=torch.bfloat16)
    primals_3 = rand_strided((2048, 2048), (2048, 1), device='cuda:1', dtype=torch.bfloat16)
    primals_4 = rand_strided((512, 2048), (2048, 1), device='cuda:1', dtype=torch.bfloat16)
    primals_5 = rand_strided((512, 2048), (2048, 1), device='cuda:1', dtype=torch.bfloat16)
    primals_6 = rand_strided((4096, 32), (32, 1), device='cuda:1', dtype=torch.complex64)
    primals_7 = rand_strided((2048, 2048), (2048, 1), device='cuda:1', dtype=torch.bfloat16)
    primals_8 = rand_strided((2048, ), (1, ), device='cuda:1', dtype=torch.bfloat16)
    primals_9 = rand_strided((8192, 2048), (2048, 1), device='cuda:1', dtype=torch.bfloat16)
    primals_10 = rand_strided((8192, 2048), (2048, 1), device='cuda:1', dtype=torch.bfloat16)
    primals_11 = rand_strided((2048, 8192), (8192, 1), device='cuda:1', dtype=torch.bfloat16)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
