use bulletformat::{ChessBoard, chess::MarlinFormat};

pub trait OutputBuckets<T>: Send + Sync + Copy + Default + 'static {
    const BUCKETS: usize;

    fn bucket(&self, pos: &T) -> u8;
}

#[deprecated(note = "You do not need to specify this anymore, it is the default!")]
#[derive(Clone, Copy, Default)]
pub struct Single;

#[allow(deprecated)]
impl<T: 'static> OutputBuckets<T> for Single {
    const BUCKETS: usize = 1;

    fn bucket(&self, _: &T) -> u8 {
        0
    }
}

#[derive(Clone, Copy, Default)]
pub struct MaterialCount<const N: usize>;
impl<const N: usize> OutputBuckets<ChessBoard> for MaterialCount<N> {
    const BUCKETS: usize = N;

    fn bucket(&self, pos: &ChessBoard) -> u8 {
        let divisor = 32usize.div_ceil(N);
        (pos.occ().count_ones() as u8 - 2) / divisor as u8
    }
}

impl<const N: usize> OutputBuckets<MarlinFormat> for MaterialCount<N> {
    const BUCKETS: usize = N;

    fn bucket(&self, pos: &MarlinFormat) -> u8 {
        let divisor = 32usize.div_ceil(N);
        (pos.occ().count_ones() as u8 - 2) / divisor as u8
    }
}

#[derive(Clone, Copy, Default)]
pub struct Phase<const N: usize>;
impl<const N: usize> OutputBuckets<ChessBoard> for Phase<N> {
    const BUCKETS: usize = N;

    fn bucket(&self, pos: &ChessBoard) -> u8 {
        const PHASE: [u32; 6] = [0, 1, 1, 3, 6, 0];
        const MAX_PHASE: u32 = 16;
        const DIVISOR : u32 = 32.div_ceil(N) as u32;

        let mut phase = 0u32;

        for (piece, _) in pos.into_iter() {
            phase += PHASE[piece & 0b0111];
        }

        phase.min(MAX_PHASE)


        (phase / divisor) as u8
    }
}

impl<const N: usize> OutputBuckets<MarlinFormat> for Phase<N> {
    const BUCKETS: usize = N;

    fn bucket(&self, pos: &MarlinFormat) -> u8 {
        const PHASE: [u32; 6] = [0, 1, 1, 3, 6, 0];
        const MAX_PHASE: u32 = 16;
        const DIVISOR : u32 = 32.div_ceil(N) as u32;

        let mut phase = 0u32;

        for (piece, _) in pos.into_iter() {
            phase += PHASE[piece & 0b0111];
        }

        phase.min(MAX_PHASE)


        (phase / divisor) as u8
    }
}
