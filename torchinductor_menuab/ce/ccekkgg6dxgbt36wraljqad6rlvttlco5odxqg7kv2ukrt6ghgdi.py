# AOT ID: ['0_backward']
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


# kernel path: /auto/home/menuab/code/YNNtitan/torchinductor_menuab/gm/cgmk4f2vea5bnjm44op3caax4jd6ltnmifw45hjme36pgromsq65.py
# Topologically Sorted Source Nodes: [silu], Original ATen: [aten.silu, aten.mul, aten.sigmoid, aten.fill, aten.sub, aten.add]
# Source node to ATen node mapping:
#   silu => convert_element_type_18, convert_element_type_19, mul_6, sigmoid
# Graph fragment:
#   %convert_element_type_18 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_20, torch.float32), kwargs = {})
#   %sigmoid : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_18,), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_18, %sigmoid), kwargs = {})
#   %convert_element_type_19 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_6, torch.bfloat16), kwargs = {})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_26, %convert_element_type_19), kwargs = {})
#   %mul_9 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_26, %view_22), kwargs = {})
#   %sigmoid_1 : [num_users=2] = call_function[target=torch.ops.aten.sigmoid.default](args = (%view_20,), kwargs = {})
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([4, 2048, 8192], 1), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:2, pin_memory: False})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%full_default, %sigmoid_1), kwargs = {})
#   %mul_10 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_20, %sub), kwargs = {})
#   %add_4 : [num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mul_10, 1), kwargs = {})
#   %mul_11 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sigmoid_1, %add_4), kwargs = {})
#   %mul_12 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_9, %mul_11), kwargs = {})
triton_poi_fused_add_fill_mul_sigmoid_silu_sub_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[67108864], 
    filename=__file__,
    triton_meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: '*bf16', 5: 'i32'}, 'device': DeviceProperties(type='cuda', index=2, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_fill_mul_sigmoid_silu_sub_0', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '54E46422D5DB2E55B804C8E038A4A0E2ECEED6FCC5402DED453936C14F5DFA13', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 67108864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x0), None).to(tl.float32)
    tmp7 = tl.load(in_ptr2 + (x0), None).to(tl.float32)
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tl.sigmoid(tmp2)
    tmp4 = tmp2 * tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp0 * tmp5
    tmp8 = tmp0 * tmp7
    tmp9 = tl.sigmoid(tmp1)
    tmp10 = 1.0
    tmp11 = tmp10 - tmp9
    tmp12 = tmp1 * tmp11
    tmp13 = tmp12 + tmp10
    tmp14 = tmp9 * tmp13
    tmp15 = tmp8 * tmp14
    tl.store(out_ptr0 + (x0), tmp6, None)
    tl.store(out_ptr1 + (x0), tmp15, None)
''', device_str='cuda')


# kernel path: /auto/home/menuab/code/YNNtitan/torchinductor_menuab/2m/c2mzpyy26bc74eftnlgk5nyal5g6wtp4adhxkpbfzmcbyqq7fpw7.py
# Topologically Sorted Source Nodes: [h, float_4], Original ATen: [aten.add, aten._to_copy, aten.mul, aten.sum, aten.div, aten.pow]
# Source node to ATen node mapping:
#   float_4 => convert_element_type_14
#   h => add_1
# Graph fragment:
#   %add_5 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_28, %view_30), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_2, %view_18), kwargs = {})
#   %convert_element_type_14 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_1, torch.float32), kwargs = {})
#   %mul_14 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_5, %primals_8), kwargs = {})
#   %convert_element_type_36 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_14, torch.float32), kwargs = {})
#   %mul_15 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_36, %convert_element_type_14), kwargs = {})
#   %mul_16 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_36, %rsqrt_1), kwargs = {})
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_15, [2], True), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Scalar](args = (%expand_2, 2048), kwargs = {})
#   %pow_4 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_14, 1.0), kwargs = {})
#   %mul_19 : [num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%pow_4, 2.0), kwargs = {})
#   %mul_20 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div, %mul_19), kwargs = {})
#   %add_6 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_16, %mul_20), kwargs = {})
#   %convert_element_type_37 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_6, torch.bfloat16), kwargs = {})
#   %add_7 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%tangents_1, %convert_element_type_37), kwargs = {})
triton_red_fused__to_copy_add_div_mul_pow_sum_1 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: '*bf16', 5: '*bf16', 6: '*fp32', 7: '*bf16', 8: 'i32', 9: 'i32'}, 'device': DeviceProperties(type='cuda', index=2, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_div_mul_pow_sum_1', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 12, 'num_reduction': 1, 'backend_hash': '54E46422D5DB2E55B804C8E038A4A0E2ECEED6FCC5402DED453936C14F5DFA13', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp12 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (2048*x0)), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r1 + (2048*x0)), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp3 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp6 = tl.load(in_ptr3 + (r1 + (2048*x0)), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp7 = tl.load(in_ptr4 + (r1 + (2048*x0)), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 * tmp3
        tmp5 = tmp4.to(tl.float32)
        tmp8 = tmp6 + tmp7
        tmp9 = tmp8.to(tl.float32)
        tmp10 = tmp5 * tmp9
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(rmask, tmp13, _tmp12)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tmp21 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp14 = tl.load(in_ptr5 + (r1 + (2048*x0)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp15 = tl.load(in_ptr0 + (r1 + (2048*x0)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp16 = tl.load(in_ptr1 + (r1 + (2048*x0)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp18 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp30 = tl.load(in_ptr3 + (r1 + (2048*x0)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp31 = tl.load(in_ptr4 + (r1 + (2048*x0)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp17 = tmp15 + tmp16
        tmp19 = tmp17 * tmp18
        tmp20 = tmp19.to(tl.float32)
        tmp22 = tmp20 * tmp21
        tmp23 = -0.5
        tmp24 = tmp12 * tmp23
        tmp25 = tmp21 * tmp21
        tmp26 = tmp25 * tmp21
        tmp27 = tmp24 * tmp26
        tmp28 = 0.00048828125
        tmp29 = tmp27 * tmp28
        tmp32 = tmp30 + tmp31
        tmp33 = tmp32.to(tl.float32)
        tmp34 = 2.0
        tmp35 = tmp33 * tmp34
        tmp36 = tmp29 * tmp35
        tmp37 = tmp22 + tmp36
        tmp38 = tmp37.to(tl.float32)
        tmp39 = tmp14 + tmp38
        tl.store(out_ptr1 + (r1 + (2048*x0)), tmp39, rmask)
''', device_str='cuda')


# kernel path: /auto/home/menuab/code/YNNtitan/torchinductor_menuab/ax/caxazbnfvms6ilx53edhbwfomi3plfgncsbhun2lduvuqprsrop6.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
# Source node to ATen node mapping:
# Graph fragment:
#   %sum_3 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_35, [3], True), kwargs = {})
triton_poi_fused_sum_2 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[4194304], 
    filename=__file__,
    triton_meta={'signature': {0: '*bf16', 1: '*bf16', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=2, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_sum_2', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': '54E46422D5DB2E55B804C8E038A4A0E2ECEED6FCC5402DED453936C14F5DFA13', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 64
    x1 = (xindex // 64)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (256*x1)), None).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (64 + x0 + (256*x1)), None).to(tl.float32)
    tmp3 = tl.load(in_ptr0 + (128 + x0 + (256*x1)), None).to(tl.float32)
    tmp5 = tl.load(in_ptr0 + (192 + x0 + (256*x1)), None).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tl.store(out_ptr0 + (x2), tmp6, None)
''', device_str='cuda')


# kernel path: /auto/home/menuab/code/YNNtitan/torchinductor_menuab/pw/cpwahkcbsocvyvnnfz4h2i67vgnmbgxezkk73gplb6oc2f4serqi.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.view_as_complex]
# Source node to ATen node mapping:
# Graph fragment:
#   %view_as_complex_2 : [num_users=1] = call_function[target=torch.ops.aten.view_as_complex.default](args = (%view_37,), kwargs = {})
triton_poi_fused_view_as_complex_3 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[4194304], 
    filename=__file__,
    triton_meta={'signature': {0: '*bf16', 1: '*fp32', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=2, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_view_as_complex_3', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': '54E46422D5DB2E55B804C8E038A4A0E2ECEED6FCC5402DED453936C14F5DFA13', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 64
    x1 = (xindex // 64)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (256*x1)), None).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (64 + x0 + (256*x1)), None).to(tl.float32)
    tmp3 = tl.load(in_ptr0 + (128 + x0 + (256*x1)), None).to(tl.float32)
    tmp5 = tl.load(in_ptr0 + (192 + x0 + (256*x1)), None).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp7 = tmp6.to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp7, None)
''', device_str='cuda')


# kernel path: /auto/home/menuab/code/YNNtitan/torchinductor_menuab/ey/ceyb234wiwz7mjlegtobalawhgwzoswfhmmh7vgaoqzdbiwqmqvx.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.view]
# Source node to ATen node mapping:
# Graph fragment:
#   %view_46 : [num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%view_42, [8192, 512]), kwargs = {})
triton_poi_fused_view_4 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[4194304], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*bf16', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=2, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_view_4', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '54E46422D5DB2E55B804C8E038A4A0E2ECEED6FCC5402DED453936C14F5DFA13', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /auto/home/menuab/code/YNNtitan/torchinductor_menuab/is/cis4wmj2muzfadeidjoxqdr3lhlodtaelgeefgxggwwzc6szyqdu.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.view_as_complex]
# Source node to ATen node mapping:
# Graph fragment:
#   %view_as_complex_3 : [num_users=1] = call_function[target=torch.ops.aten.view_as_complex.default](args = (%view_38,), kwargs = {})
triton_poi_fused_view_as_complex_5 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[16777216], 
    filename=__file__,
    triton_meta={'signature': {0: '*bf16', 1: '*fp32', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=2, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_view_as_complex_5', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '54E46422D5DB2E55B804C8E038A4A0E2ECEED6FCC5402DED453936C14F5DFA13', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
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


# kernel path: /auto/home/menuab/code/YNNtitan/torchinductor_menuab/x7/cx7plcck4dg6eq7wy5ejk4kdzbonbik5jnzctuou662j5dhozwq7.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.view]
# Source node to ATen node mapping:
# Graph fragment:
#   %view_48 : [num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%view_43, [8192, 2048]), kwargs = {})
triton_poi_fused_view_6 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[16777216], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*bf16', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=2, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_view_6', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '54E46422D5DB2E55B804C8E038A4A0E2ECEED6FCC5402DED453936C14F5DFA13', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
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


# kernel path: /auto/home/menuab/code/YNNtitan/torchinductor_menuab/tt/ctt6lu6opqtcbdb5c4cz6ozxmnkmunovcdwh7k5pthgd7fx7tvv3.py
# Topologically Sorted Source Nodes: [h, float_4, mul_4, output_4, float_1, mul, output], Original ATen: [aten.add, aten._to_copy, aten.mul, aten.sum]
# Source node to ATen node mapping:
#   float_1 => convert_element_type
#   float_4 => convert_element_type_14
#   h => add_1
#   mul => mul
#   mul_4 => mul_4
#   output => convert_element_type_1
#   output_4 => convert_element_type_15
# Graph fragment:
#   %add_5 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_28, %view_30), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_2, %view_18), kwargs = {})
#   %convert_element_type_14 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_1, torch.float32), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_14, %rsqrt_1), kwargs = {})
#   %convert_element_type_15 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_4, torch.bfloat16), kwargs = {})
#   %mul_13 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_5, %convert_element_type_15), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_13, [0, 1], True), kwargs = {})
#   %add_8 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_45, %view_47), kwargs = {})
#   %add_9 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_8, %view_49), kwargs = {})
#   %convert_element_type : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_2, torch.float32), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type, %rsqrt), kwargs = {})
#   %convert_element_type_1 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul, torch.bfloat16), kwargs = {})
#   %mul_23 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_9, %convert_element_type_1), kwargs = {})
#   %sum_5 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_23, [0, 1], True), kwargs = {})
triton_red_fused__to_copy_add_mul_sum_7 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.reduction(
    size_hints=[131072, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: '*fp32', 5: '*bf16', 6: '*bf16', 7: '*bf16', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32', 12: 'i32'}, 'device': DeviceProperties(type='cuda', index=2, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_mul_sum_7', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 9, 'num_reduction': 2, 'backend_hash': '54E46422D5DB2E55B804C8E038A4A0E2ECEED6FCC5402DED453936C14F5DFA13', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 131072
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 2048
    x1 = (xindex // 2048)
    _tmp12 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp25 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (2048*r2) + (262144*x1)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (x0 + (2048*r2) + (262144*x1)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp3 = tl.load(in_ptr2 + (x0 + (2048*r2) + (262144*x1)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp4 = tl.load(in_ptr3 + (x0 + (2048*r2) + (262144*x1)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp7 = tl.load(in_ptr4 + (r2 + (128*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp14 = tl.load(in_ptr5 + (x0 + (2048*r2) + (262144*x1)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp15 = tl.load(in_ptr6 + (x0 + (2048*r2) + (262144*x1)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp17 = tl.load(in_ptr7 + (x0 + (2048*r2) + (262144*x1)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp20 = tl.load(in_ptr8 + (r2 + (128*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp5 = tmp3 + tmp4
        tmp6 = tmp5.to(tl.float32)
        tmp8 = tmp6 * tmp7
        tmp9 = tmp8.to(tl.float32)
        tmp10 = tmp2 * tmp9
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(rmask, tmp13, _tmp12)
        tmp16 = tmp14 + tmp15
        tmp18 = tmp16 + tmp17
        tmp19 = tmp3.to(tl.float32)
        tmp21 = tmp19 * tmp20
        tmp22 = tmp21.to(tl.float32)
        tmp23 = tmp18 * tmp22
        tmp24 = tl.broadcast_to(tmp23, [XBLOCK, RBLOCK])
        tmp26 = _tmp25 + tmp24
        _tmp25 = tl.where(rmask, tmp26, _tmp25)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp12, None)
    tmp25 = tl.sum(_tmp25, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp25, None)
''', device_str='cuda')


# kernel path: /auto/home/menuab/code/YNNtitan/torchinductor_menuab/qy/cqypnnha6mjaft46am2rkocnboecmqmc6ccbbxbaulhyyf5skjv7.py
# Topologically Sorted Source Nodes: [h, float_4, mul_4, output_4], Original ATen: [aten.add, aten._to_copy, aten.mul, aten.sum]
# Source node to ATen node mapping:
#   float_4 => convert_element_type_14
#   h => add_1
#   mul_4 => mul_4
#   output_4 => convert_element_type_15
# Graph fragment:
#   %add_5 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_28, %view_30), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_2, %view_18), kwargs = {})
#   %convert_element_type_14 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_1, torch.float32), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_14, %rsqrt_1), kwargs = {})
#   %convert_element_type_15 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_4, torch.bfloat16), kwargs = {})
#   %mul_13 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_5, %convert_element_type_15), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_13, [0, 1], True), kwargs = {})
triton_per_fused__to_copy_add_mul_sum_8 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[2048, 64],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*bf16', 2: 'i32', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=2, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_mul_sum_8', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '54E46422D5DB2E55B804C8E038A4A0E2ECEED6FCC5402DED453936C14F5DFA13', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (2048*r1)), None)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.sum(tmp1, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp3, None)
''', device_str='cuda')


# kernel path: /auto/home/menuab/code/YNNtitan/torchinductor_menuab/ey/ceyq7s2m7eg7wsbogh5g74szf4crhvboq7n4vbjwimy4hdzmjf4l.py
# Topologically Sorted Source Nodes: [float_1], Original ATen: [aten.add, aten._to_copy, aten.mul, aten.sum, aten.div, aten.pow]
# Source node to ATen node mapping:
#   float_1 => convert_element_type
# Graph fragment:
#   %add_8 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_45, %view_47), kwargs = {})
#   %add_9 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_8, %view_49), kwargs = {})
#   %convert_element_type : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_2, torch.float32), kwargs = {})
#   %mul_24 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_9, %primals_1), kwargs = {})
#   %convert_element_type_58 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_24, torch.float32), kwargs = {})
#   %mul_25 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_58, %convert_element_type), kwargs = {})
#   %mul_26 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_58, %rsqrt), kwargs = {})
#   %sum_6 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_25, [2], True), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Scalar](args = (%expand_3, 2048), kwargs = {})
#   %pow_6 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type, 1.0), kwargs = {})
#   %mul_29 : [num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%pow_6, 2.0), kwargs = {})
#   %mul_30 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_1, %mul_29), kwargs = {})
#   %add_10 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_26, %mul_30), kwargs = {})
#   %convert_element_type_59 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_10, torch.bfloat16), kwargs = {})
#   %add_11 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_7, %convert_element_type_59), kwargs = {})
triton_red_fused__to_copy_add_div_mul_pow_sum_9 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: '*bf16', 5: '*bf16', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': DeviceProperties(type='cuda', index=2, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_div_mul_pow_sum_9', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 12, 'num_reduction': 1, 'backend_hash': '54E46422D5DB2E55B804C8E038A4A0E2ECEED6FCC5402DED453936C14F5DFA13', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp12 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (2048*x0)), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r1 + (2048*x0)), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp3 = tl.load(in_ptr2 + (r1 + (2048*x0)), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp5 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp8 = tl.load(in_ptr4 + (r1 + (2048*x0)), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 + tmp3
        tmp6 = tmp4 * tmp5
        tmp7 = tmp6.to(tl.float32)
        tmp9 = tmp8.to(tl.float32)
        tmp10 = tmp7 * tmp9
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(rmask, tmp13, _tmp12)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tmp23 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp14 = tl.load(in_out_ptr0 + (r1 + (2048*x0)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp15 = tl.load(in_ptr0 + (r1 + (2048*x0)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp16 = tl.load(in_ptr1 + (r1 + (2048*x0)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp18 = tl.load(in_ptr2 + (r1 + (2048*x0)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp20 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp32 = tl.load(in_ptr4 + (r1 + (2048*x0)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp17 = tmp15 + tmp16
        tmp19 = tmp17 + tmp18
        tmp21 = tmp19 * tmp20
        tmp22 = tmp21.to(tl.float32)
        tmp24 = tmp22 * tmp23
        tmp25 = -0.5
        tmp26 = tmp12 * tmp25
        tmp27 = tmp23 * tmp23
        tmp28 = tmp27 * tmp23
        tmp29 = tmp26 * tmp28
        tmp30 = 0.00048828125
        tmp31 = tmp29 * tmp30
        tmp33 = tmp32.to(tl.float32)
        tmp34 = 2.0
        tmp35 = tmp33 * tmp34
        tmp36 = tmp31 * tmp35
        tmp37 = tmp24 + tmp36
        tmp38 = tmp37.to(tl.float32)
        tmp39 = tmp14 + tmp38
        tl.store(in_out_ptr0 + (r1 + (2048*x0)), tmp39, rmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_8, rsqrt, view, permute_3, permute_4, permute_5, getitem, getitem_1, getitem_6, getitem_7, mm_3, rsqrt_1, view_19, mm_4, mm_5, view_23, permute_13, permute_17, permute_22, permute_26, _conj, permute_34, permute_38, permute_42, tangents_1 = args
    args.clear()
    assert_size_stride(primals_1, (2048, ), (1, ))
    assert_size_stride(primals_2, (4, 2048, 2048), (4194304, 2048, 1))
    assert_size_stride(primals_8, (2048, ), (1, ))
    assert_size_stride(rsqrt, (4, 2048, 1), (2048, 1, 1))
    assert_size_stride(view, (8192, 2048), (2048, 1))
    assert_size_stride(permute_3, (4, 32, 2048, 64), (4194304, 64, 2048, 1))
    assert_size_stride(permute_4, (4, 32, 2048, 64), (4194304, 64, 2048, 1))
    assert_size_stride(permute_5, (4, 32, 2048, 64), (4194304, 64, 2048, 1))
    assert_size_stride(getitem, (4, 32, 2048, 64), (4194304, 64, 2048, 1))
    assert_size_stride(getitem_1, (4, 32, 2048), (65536, 2048, 1))
    assert_size_stride(getitem_6, (), ())
    assert_size_stride(getitem_7, (), ())
    assert_size_stride(mm_3, (8192, 2048), (2048, 1))
    assert_size_stride(rsqrt_1, (4, 2048, 1), (2048, 1, 1))
    assert_size_stride(view_19, (8192, 2048), (2048, 1))
    assert_size_stride(mm_4, (8192, 8192), (8192, 1))
    assert_size_stride(mm_5, (8192, 8192), (8192, 1))
    assert_size_stride(view_23, (8192, 8192), (8192, 1))
    assert_size_stride(permute_13, (2048, 8192), (8192, 1))
    assert_size_stride(permute_17, (8192, 2048), (2048, 1))
    assert_size_stride(permute_22, (8192, 2048), (2048, 1))
    assert_size_stride(permute_26, (2048, 2048), (2048, 1))
    assert_size_stride(_conj, (1, 2048, 1, 32), (65536, 32, 32, 1))
    assert_size_stride(permute_34, (512, 2048), (2048, 1))
    assert_size_stride(permute_38, (512, 2048), (2048, 1))
    assert_size_stride(permute_42, (2048, 2048), (2048, 1))
    assert_size_stride(tangents_1, (4, 2048, 2048), (4194304, 2048, 1))
    with torch.cuda._DeviceGuard(2):
        torch.cuda.set_device(2)
        buf0 = empty_strided_cuda((2048, 8192), (8192, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(tangents_1, (2048, 8192), (1, 2048), 0), view_23, out=buf0)
        del view_23
        buf1 = empty_strided_cuda((8192, 8192), (8192, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(tangents_1, (8192, 2048), (2048, 1), 0), permute_13, out=buf1)
        del permute_13
        buf2 = empty_strided_cuda((4, 2048, 8192), (16777216, 8192, 1), torch.bfloat16)
        buf5 = empty_strided_cuda((4, 2048, 8192), (16777216, 8192, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [silu], Original ATen: [aten.silu, aten.mul, aten.sigmoid, aten.fill, aten.sub, aten.add]
        stream2 = get_raw_stream(2)
        triton_poi_fused_add_fill_mul_sigmoid_silu_sub_0.run(buf1, mm_4, mm_5, buf2, buf5, 67108864, grid=grid(67108864), stream=stream2)
        del buf1
        del mm_4
        del mm_5
        buf3 = empty_strided_cuda((8192, 2048), (2048, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf2, (8192, 8192), (1, 8192), 0), view_19, out=buf3)
        buf4 = empty_strided_cuda((8192, 2048), (2048, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf2, (8192, 8192), (8192, 1), 0), permute_17, out=buf4)
        del buf2
        del permute_17
        buf6 = empty_strided_cuda((8192, 2048), (2048, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf5, (8192, 8192), (1, 8192), 0), view_19, out=buf6)
        del view_19
        buf7 = empty_strided_cuda((8192, 2048), (2048, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf5, (8192, 8192), (8192, 1), 0), permute_22, out=buf7)
        del buf5
        del permute_22
        buf11 = empty_strided_cuda((4, 2048, 2048), (4194304, 2048, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [h, float_4], Original ATen: [aten.add, aten._to_copy, aten.mul, aten.sum, aten.div, aten.pow]
        triton_red_fused__to_copy_add_div_mul_pow_sum_1.run(buf4, buf7, primals_8, primals_2, mm_3, tangents_1, rsqrt_1, buf11, 8192, 2048, grid=grid(8192), stream=stream2)
        del primals_8
        del tangents_1
        buf13 = empty_strided_cuda((8192, 2048), (2048, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf11, (8192, 2048), (2048, 1), 0), permute_26, out=buf13)
        del permute_26
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._scaled_dot_product_flash_attention_backward]
        buf14 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(reinterpret_tensor(buf13, (4, 32, 2048, 64), (4194304, 64, 2048, 1), 0), permute_3, permute_4, permute_5, getitem, getitem_1, None, None, 2048, 2048, 0.0, True, getitem_6, getitem_7, scale=0.125)
        del getitem_1
        del getitem_6
        del getitem_7
        del permute_3
        del permute_4
        del permute_5
        buf17 = buf14[2]
        buf32 = empty_strided_cuda((4, 2048, 8, 1, 64), (1048576, 512, 64, 64, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
        triton_poi_fused_sum_2.run(buf17, buf32, 4194304, grid=grid(4194304), stream=stream2)
        buf34 = reinterpret_tensor(buf17, (8192, 2048), (2048, 1), 0); del buf17  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf32, (8192, 512), (512, 1), 0), permute_34, out=buf34)
        del permute_34
        buf16 = buf14[1]
        buf18 = empty_strided_cuda((4, 2048, 8, 32, 2), (1048576, 512, 64, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.view_as_complex]
        triton_poi_fused_view_as_complex_3.run(buf16, buf18, 4194304, grid=grid(4194304), stream=stream2)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.view_as_complex]
        buf19 = torch.ops.aten.view_as_complex.default(buf18)
        buf20 = buf19
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul]
        buf21 = torch.ops.aten.mul.Tensor(buf20, _conj)
        del buf18
        del buf19
        del buf20
        buf22 = buf21
        del buf21
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.view_as_real]
        buf28 = torch.ops.aten.view_as_real.default(buf22)
        buf29 = buf28
        buf35 = empty_strided_cuda((8192, 512), (512, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_4.run(buf29, buf35, 4194304, grid=grid(4194304), stream=stream2)
        del buf22
        del buf28
        del buf29
        buf37 = reinterpret_tensor(buf16, (8192, 2048), (2048, 1), 0); del buf16  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf35, permute_38, out=buf37)
        del permute_38
        buf15 = buf14[0]
        del buf14
        buf23 = empty_strided_cuda((4, 2048, 32, 32, 2), (4194304, 2048, 64, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.view_as_complex]
        triton_poi_fused_view_as_complex_5.run(buf15, buf23, 16777216, grid=grid(16777216), stream=stream2)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.view_as_complex]
        buf24 = torch.ops.aten.view_as_complex.default(buf23)
        buf25 = buf24
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul]
        buf26 = torch.ops.aten.mul.Tensor(buf25, _conj)
        del _conj
        del buf23
        del buf24
        del buf25
        buf27 = buf26
        del buf26
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.view_as_real]
        buf30 = torch.ops.aten.view_as_real.default(buf27)
        buf31 = buf30
        buf38 = reinterpret_tensor(buf15, (8192, 2048), (2048, 1), 0); del buf15  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_6.run(buf31, buf38, 16777216, grid=grid(16777216), stream=stream2)
        del buf27
        del buf30
        del buf31
        buf40 = buf13; del buf13  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf38, permute_42, out=buf40)
        del permute_42
        buf8 = empty_strided_cuda((1, 1, 2048, 64), (131072, 131072, 1, 2048), torch.float32)
        buf41 = empty_strided_cuda((1, 1, 2048, 64), (131072, 131072, 1, 2048), torch.float32)
        # Topologically Sorted Source Nodes: [h, float_4, mul_4, output_4, float_1, mul, output], Original ATen: [aten.add, aten._to_copy, aten.mul, aten.sum]
        triton_red_fused__to_copy_add_mul_sum_7.run(buf4, buf7, primals_2, mm_3, rsqrt_1, buf34, buf37, buf40, rsqrt, buf8, buf41, 131072, 128, grid=grid(131072), stream=stream2)
        del buf4
        del buf7
        del mm_3
        del rsqrt_1
        buf9 = empty_strided_cuda((1, 1, 2048), (2048, 2048, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [h, float_4, mul_4, output_4], Original ATen: [aten.add, aten._to_copy, aten.mul, aten.sum]
        triton_per_fused__to_copy_add_mul_sum_8.run(buf8, buf9, 2048, 64, grid=grid(2048), stream=stream2)
        del buf8
        buf12 = empty_strided_cuda((2048, 2048), (2048, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf11, (2048, 8192), (1, 2048), 0), reinterpret_tensor(getitem, (8192, 2048), (2048, 1), 0), out=buf12)
        del getitem
        buf33 = empty_strided_cuda((512, 2048), (2048, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf32, (512, 8192), (1, 512), 0), view, out=buf33)
        del buf32
        buf36 = empty_strided_cuda((512, 2048), (2048, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf35, (512, 8192), (1, 512), 0), view, out=buf36)
        buf39 = reinterpret_tensor(buf35, (2048, 2048), (2048, 1), 0); del buf35  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf38, (2048, 8192), (1, 2048), 0), view, out=buf39)
        del buf38
        del view
        buf42 = empty_strided_cuda((1, 1, 2048), (2048, 2048, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [float_1, mul, output], Original ATen: [aten.add, aten._to_copy, aten.mul, aten.sum]
        triton_per_fused__to_copy_add_mul_sum_8.run(buf41, buf42, 2048, 64, grid=grid(2048), stream=stream2)
        del buf41
        buf44 = buf11; del buf11  # reuse
        # Topologically Sorted Source Nodes: [float_1], Original ATen: [aten.add, aten._to_copy, aten.mul, aten.sum, aten.div, aten.pow]
        triton_red_fused__to_copy_add_div_mul_pow_sum_9.run(buf44, buf34, buf37, buf40, primals_1, primals_2, rsqrt, 8192, 2048, grid=grid(8192), stream=stream2)
        del buf34
        del buf37
        del buf40
        del primals_1
        del primals_2
        del rsqrt
    return (reinterpret_tensor(buf42, (2048, ), (1, ), 0), buf44, buf39, buf36, buf33, None, buf12, reinterpret_tensor(buf9, (2048, ), (1, ), 0), buf6, buf3, buf0, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((2048, ), (1, ), device='cuda:2', dtype=torch.bfloat16)
    primals_2 = rand_strided((4, 2048, 2048), (4194304, 2048, 1), device='cuda:2', dtype=torch.bfloat16)
    primals_8 = rand_strided((2048, ), (1, ), device='cuda:2', dtype=torch.bfloat16)
    rsqrt = rand_strided((4, 2048, 1), (2048, 1, 1), device='cuda:2', dtype=torch.float32)
    view = rand_strided((8192, 2048), (2048, 1), device='cuda:2', dtype=torch.bfloat16)
    permute_3 = rand_strided((4, 32, 2048, 64), (4194304, 64, 2048, 1), device='cuda:2', dtype=torch.bfloat16)
    permute_4 = rand_strided((4, 32, 2048, 64), (4194304, 64, 2048, 1), device='cuda:2', dtype=torch.bfloat16)
    permute_5 = rand_strided((4, 32, 2048, 64), (4194304, 64, 2048, 1), device='cuda:2', dtype=torch.bfloat16)
    getitem = rand_strided((4, 32, 2048, 64), (4194304, 64, 2048, 1), device='cuda:2', dtype=torch.bfloat16)
    getitem_1 = rand_strided((4, 32, 2048), (65536, 2048, 1), device='cuda:2', dtype=torch.float32)
    getitem_6 = rand_strided((), (), device='cuda:2', dtype=torch.int64)
    getitem_7 = rand_strided((), (), device='cuda:2', dtype=torch.int64)
    mm_3 = rand_strided((8192, 2048), (2048, 1), device='cuda:2', dtype=torch.bfloat16)
    rsqrt_1 = rand_strided((4, 2048, 1), (2048, 1, 1), device='cuda:2', dtype=torch.float32)
    view_19 = rand_strided((8192, 2048), (2048, 1), device='cuda:2', dtype=torch.bfloat16)
    mm_4 = rand_strided((8192, 8192), (8192, 1), device='cuda:2', dtype=torch.bfloat16)
    mm_5 = rand_strided((8192, 8192), (8192, 1), device='cuda:2', dtype=torch.bfloat16)
    view_23 = rand_strided((8192, 8192), (8192, 1), device='cuda:2', dtype=torch.bfloat16)
    permute_13 = rand_strided((2048, 8192), (8192, 1), device='cuda:2', dtype=torch.bfloat16)
    permute_17 = rand_strided((8192, 2048), (2048, 1), device='cuda:2', dtype=torch.bfloat16)
    permute_22 = rand_strided((8192, 2048), (2048, 1), device='cuda:2', dtype=torch.bfloat16)
    permute_26 = rand_strided((2048, 2048), (2048, 1), device='cuda:2', dtype=torch.bfloat16)
    _conj = rand_strided((1, 2048, 1, 32), (65536, 32, 32, 1), device='cuda:2', dtype=torch.complex64)
    permute_34 = rand_strided((512, 2048), (2048, 1), device='cuda:2', dtype=torch.bfloat16)
    permute_38 = rand_strided((512, 2048), (2048, 1), device='cuda:2', dtype=torch.bfloat16)
    permute_42 = rand_strided((2048, 2048), (2048, 1), device='cuda:2', dtype=torch.bfloat16)
    tangents_1 = rand_strided((4, 2048, 2048), (4194304, 2048, 1), device='cuda:2', dtype=torch.bfloat16)
    fn = lambda: call([primals_1, primals_2, primals_8, rsqrt, view, permute_3, permute_4, permute_5, getitem, getitem_1, getitem_6, getitem_7, mm_3, rsqrt_1, view_19, mm_4, mm_5, view_23, permute_13, permute_17, permute_22, permute_26, _conj, permute_34, permute_38, permute_42, tangents_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
