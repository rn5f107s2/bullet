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

#[derive(Clone, Copy)]
pub struct KingOutputBuckets {
    layout: [usize; 64],
}

impl Default for KingOutputBuckets {
    fn default() -> Self {
        Self {
            layout: [0; 64],
        }
    }
}

impl KingOutputBuckets {
    pub fn new(bucket_layout: [usize; 64]) -> Self {
        Self { layout: bucket_layout }
    }
}

impl OutputBuckets<ChessBoard> for KingOutputBuckets {
    const BUCKETS: usize = 15;

    fn bucket(&self, pos: &ChessBoard) -> u8 {
        (self.layout[pos.our_ksq() as usize] as u8 | (self.layout[pos.opp_ksq()] << 4) as u8)
    }
}