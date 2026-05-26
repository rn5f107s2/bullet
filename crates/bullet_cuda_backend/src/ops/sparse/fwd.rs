use acyclib::device::{function, operation::DiffableFromOutput};

use crate::{
    CudaDevice,
    kernel::{Expr, Kernel, KernelArgs, KernelInput},
};

const MAXIMUM_BLOCKS_Y: u32 = 32768;

const N: u32 = 32;
const HL: u32 = N * 64;

pub fn kernel(desc: function::SparseAffineActivate<CudaDevice>) -> Kernel {
    let output_shape = desc.weights_shape * desc.input_shape;
    let indices = desc.indices;

    assert_eq!(desc.weights_shape.size(), desc.weights.shape().size());
    assert_eq!(desc.input_shape.size(), indices.shape().size());
    assert_eq!(desc.input_shape.cols(), 1);
    assert_eq!(output_shape.cols(), 1);

    let bias = desc.biases.as_ref().map(|x| x.batch_size().is_some());

    let batched = indices.batch_size().is_some();
    let nnz = indices.sparse().nnz();
    let m = output_shape.rows();
    let vectorise = true;

    let code = kernel_str(bias, nnz, m, desc.activation, vectorise);

    let batch_size = Expr::Var;

    let mut inputs = vec![
        KernelInput::Size(batch_size.clone()),
        KernelInput::Slice {
            slice: desc.weights,
            layout: None,
            mutable: false,
            batched: false,
            shape: desc.weights_shape,
        },
        KernelInput::Slice { slice: indices, layout: Some(nnz), mutable: false, batched, shape: desc.input_shape },
        KernelInput::Slice { slice: desc.output, layout: None, mutable: true, batched, shape: output_shape },
    ];

    if let Some(bias) = desc.biases {
        let batched = bias.batch_size().is_some();
        let shape = bias.shape();
        assert_eq!(shape.size(), output_shape.size());

        inputs.push(KernelInput::Slice { slice: bias, layout: None, mutable: false, batched, shape: output_shape });
    }

    const MAXIMUM_BLOCKS_Y: Expr<i32> = Expr::Const(32768);

    let (chunks, threads, smem) = if vectorise {
        let m4 = m / 4;
        let threads = m4.min(1024);
        let chunks = m4.div_ceil(threads);
        (chunks, threads, 4 * nnz as u32)
    } else {
        let threads = m.min(1024);
        let chunks = m.div_ceil(threads);
        (chunks, threads, 0)
    };

    let ky = batch_size.min(&MAXIMUM_BLOCKS_Y);
    let kz = (batch_size + MAXIMUM_BLOCKS_Y - 1) / MAXIMUM_BLOCKS_Y;
    let grid_dim = [Expr::Const(chunks as i32), ky, kz];
    let block_dim = [Expr::Const(threads as i32), Expr::Const(1), Expr::Const(1)];
    let shared_mem_bytes = Expr::Const(smem as i32);

    let args = KernelArgs { inputs, grid_dim, block_dim, shared_mem_bytes };

    unsafe { Kernel::new("SparseAffineActiveBackward".to_string(), code, args).unwrap() }
}

fn act_str(act: DiffableFromOutput) -> &'static str {
    match act {
        DiffableFromOutput::Identity => "x",
        DiffableFromOutput::ReLU => "x > 0.0F ? x : 0.0F",
        DiffableFromOutput::CReLU => "x < 0.0F ? 0.0F : (x > 1.0F ? 1.0F : x)",
        DiffableFromOutput::SCReLU => "x < 0.0F ? 0.0F : (x > 1.0F ? 1.0F : (x * x))",
        DiffableFromOutput::SqrReLU => "x < 0.0F ? 0.0F : (x * x)",
        DiffableFromOutput::Sigmoid => "1.0F / (1.0F + expf(-x))",
    }
}

fn kernel_str(bias: Option<bool>, nnz: usize, m: usize, activation: DiffableFromOutput, vectorise: bool) -> String {
    let op = format!("__device__ float op(float x) {{ return {}; }}", act_str(activation));

    let code = if vectorise { vectorised_kernel(bias) } else { fallback_kernel(bias) };

    let bias_args = if bias.is_some() { ", const float* B" } else { "" };

    format!(
        "
        constexpr int MaximumBlocksY = {MAXIMUM_BLOCKS_Y};

        {op}

        extern \"C\" __global__ void kernel(
            const int k,
            const float* A,
            const int* X,
            float* Y{bias_args})
        {{
            constexpr int m = {m};
            constexpr int nnz = {nnz};
            const int loc = MaximumBlocksY * blockIdx.z + blockIdx.y;
            const int row = blockIdx.x * blockDim.x + threadIdx.x;
            {code}
        }}"
    )
}

// Claude slop
fn vectorised_kernel(_bias: Option<bool>) -> String {
    // One thread covers 4 consecutive output elements via float4 loads/stores.
    // Requires: m % 4 == 0, N % 4 == 0, caller launches m/4 threads in x.
    format!(
        "
        // Each thread owns a group of 4 consecutive rows
        if (row >= m / 4 || loc >= k) return;
        const int row4 = row * 4;  // first scalar row this thread owns

        // --- Vectorised zero-out (128-bit store) ---
        reinterpret_cast<float4*>(Y + m * loc)[row] = make_float4(0.f, 0.f, 0.f, 0.f);

        if (row4 >= nnz * {N})
            return;

        // Index derivation is identical to the scalar kernel, but for the
        // first element of the group; the remaining three are at +1/+2/+3.
        float4 sum = make_float4(0.0F, 0.0F, 0.0F, 0.0F);
        const int featIdx = row4 / {N};
        const int feat    = X[nnz * loc + featIdx];
        if (feat == -1)
            return;

        const int pc   = feat / 64;
        const int sq   = feat % 64;
        const int idx  = row4 % {N};   // always 4-aligned because N%4==0
        const int flip = 7 * !!(sq & 4);
        const int base = pc * {HL} + idx;
        const int nRow   = base + (sq ^ flip) * {N};  // load  base (4-aligned)
        const int outRow = base + sq * {N};            // store base (4-aligned)

        // --- Inner accumulation loop with vectorised 128-bit loads ---
        // A[j*m + nRow .. nRow+3] are contiguous floats → single float4 load
        for (int i = 0; i < nnz; i++) {{
            const int j = X[nnz * loc + i] ^ flip;
            if (j == -1) break;
            const float4 a = reinterpret_cast<const float4*>(A + j * m + nRow)[0];
            sum.x += a.x;
            sum.y += a.y;
            sum.z += a.z;
            sum.w += a.w;
        }}

        // Apply activation element-wise
        sum.x = op(sum.x);
        sum.y = op(sum.y);
        sum.z = op(sum.z);
        sum.w = op(sum.w);

        // --- Vectorised store (128-bit) ---
        // Y[m*loc + outRow .. outRow+3] are contiguous → single float4 store
        reinterpret_cast<float4*>(Y + m * loc + outRow)[0] = sum;"
    )
}

fn fallback_kernel(_bias: Option<bool>) -> String {
    format!(
        "
        if (row >= m || loc >= k) return;

        Y[m * loc + row] = 0;

        if (row >= nnz * {N})
            return;

        float sum = 0.0F;

        const int featIdx = row / {N};
        const int feat    = X[nnz * loc + featIdx];

        if (feat == -1)
            return;

        const int pc = feat / 64;
        const int sq = feat % 64;

        const int idx = row % {N};

        const int flip = 7 * !!(sq & 4);

        const int base = pc * {HL} + idx;
        const int nRow   = base + (sq ^ flip) * {N};
        const int outRow = base + sq * {N};

        for (int i = 0; i < nnz; i++) {{
            const int j = X[nnz * loc + i] ^ flip;

            if (j == -1) break;

            sum += A[j * m + nRow];
        }}

        Y[m * loc + outRow] = op(sum);"
    )
}
