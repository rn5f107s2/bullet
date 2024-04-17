use bulletformat::{BulletFormat, ChessBoard};

pub trait OutputBuckets<T: BulletFormat>: Send + Sync + Copy + Default + 'static {
    const BUCKETS: usize;

    fn bucket(&self, pos: &T) -> u8;
}

#[derive(Clone, Copy, Default)]
pub struct Single;
impl<T: BulletFormat + 'static> OutputBuckets<T> for Single {
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
        let divisor = (32 + N - 1) / N;
        (pos.occ().count_ones() as u8 - 2) / divisor as u8
    }
}

#[derive(Clone, Copy, Default)]
pub struct OCB;
impl OutputBuckets<ChessBoard> for OCB {
    const BUCKETS: usize = 2;

    fn bucket(&self, pos: &ChessBoard) -> u8 {
        let bishop: u8 = 2;
        
        let mut iter = pos.into_iter() 
                          .filter(|(pc, _sq)| (pc & 0b0111) == bishop)
                          .map(|(pc, sq)| { ( (pc & 0b1000) >> 3, ((sq & 1) ^ ((sq & 8) >> 3))) });

        let Some((pc1, sq1)) = iter.next() else { return 0 };
        let Some((pc2, sq2)) = iter.next() else { return 0 };
        let b3 = iter.next();
        
        if pc1 != pc2 && sq1 != sq2 && b3 == None { 1 } else { 0 }
    }
}
