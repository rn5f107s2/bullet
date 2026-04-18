use bulletformat::{ChessBoard, chess::MarlinFormat};

use crate::game::inputs::get_num_buckets;

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

pub struct KingOutputBuckets<const LAYOUT: [usize; 64]>;
impl<const N: usize> OutputBuckets<ChessBoard> for KingOutputBuckets<N> {
    const BUCKETS: usize = get_num_buckets(LAYOUT);

    fn bucket(&self, pos: &ChessBoard) -> u8 {
        LAYOUT[pos.our_ksq()]
    }
}