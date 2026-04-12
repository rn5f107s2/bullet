use acyclib::device::{function, operation::DiffableFromOutput};

use crate::{
    CudaDevice,
    kernel::{Expr, Kernel, KernelArgs, KernelInput},
};

const MAXIMUM_BLOCKS_Y: u32 = 32768;

const N: u32 = 16;
const HL: u32 = 64 * N;

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
    let vectorise = m % 4 == 0 && m >= 128;

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

    let chunks = m.div_ceil(1024);
    let threads = if chunks == 1 { m } else { 1024 };

    let grid_dim = [Expr::Const(chunks as i32), batch_size, Expr::Const(1)];
    let block_dim: [Expr<i32>; 3] = [Expr::Const(threads as i32), Expr::Const(1), Expr::Const(1)];
    let shared_mem_bytes: Expr<i32> = Expr::Const(0);

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

    let code = fallback_kernel(bias);

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
            const size_t elem = blockIdx.x * blockDim.x + threadIdx.x;
            {code}
        }}"
    )
}

fn fallback_kernel(_bias: Option<bool>) -> String {
    format!(
        "
        if (elem >= m) return;

        *(Y + 2 * m * blockIdx.y + elem) = 0;

        if (elem >= k * {N})
            return;

        int index = elem / {N};

        const int* thisInput = X + k * blockIdx.y;

        const int feat = X[index];

        if (feat == -1) return;

        const int featSqOur = feat % 64;
        const int featPcOur = feat / 64;

        const int ourIndex = featPcOur * {HL} + featSqOur * {N} + elem % {N};

        float sum = 0.0F;

        float* ourOutput = Y + outputSize * blockIdx.y + ourIndex;

        for (int i = 0; i < nnz; i++) {{
            const int j = thisInput[i];

            if (j == -1) break;

            const size_t ourIdx = static_cast<size_t>(j) * {HL} * 12 + ourIndex;
        
            sum += A[ourIdx];
        }}

        *ourOutput = op(sum);"
    )
}
