use std::mem::MaybeUninit;

use super::{Bitboard, MAX_MOVES, Move, MoveKind, Square};

#[derive(Copy, Clone)]
pub struct MoveEntry {
    pub mv: Move,
    pub score: i32,
}

pub struct MoveList {
    moves: [MaybeUninit<Move>; MAX_MOVES],
    scores: [MaybeUninit<i32>; MAX_MOVES],
    len: usize,
}

impl MoveList {
    pub const fn new() -> Self {
        let moves = unsafe { MaybeUninit::uninit().assume_init() };
        let scores = unsafe { MaybeUninit::uninit().assume_init() };
        Self { moves, scores, len: 0 }
    }

    pub const fn len(&self) -> usize {
        self.len
    }

    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn push(&mut self, from: Square, to: Square, kind: MoveKind) {
        self.moves[self.len] = MaybeUninit::new(Move::new(from, to, kind));
        self.len += 1;
    }

    #[cfg(not(target_feature = "avx512vbmi2"))]
    pub fn push_setwise(&mut self, from: Square, to_bb: Bitboard, kind: MoveKind) {
        for to in to_bb {
            self.push(from, to, kind);
        }
    }

    #[cfg(target_feature = "avx512vbmi2")]
    pub fn push_setwise(&mut self, from: Square, to_bb: Bitboard, kind: MoveKind) {
        if !to_bb.is_empty() {
            use std::{arch::x86_64::*, mem::transmute};

            unsafe {
                let template0: __m512i = transmute({
                    let mut template0: [Move; 32] = [Move::NULL; 32];
                    for (i, e) in template0.iter_mut().enumerate() {
                        *e = Move::new(Square::new(0u8), Square::new(i as u8), transmute::<u8, MoveKind>(0u8));
                    }
                    template0
                });
                let template1: __m512i = transmute({
                    let mut template1: [Move; 32] = [Move::NULL; 32];
                    for (i, e) in template1.iter_mut().enumerate() {
                        *e = Move::new(Square::new(0u8), Square::new(32 + i as u8), transmute::<u8, MoveKind>(0u8));
                    }
                    template1
                });

                let extra = _mm512_set1_epi16(transmute::<Move, i16>(Move::new(from, Square::new(0u8), kind)));

                self.splat16(to_bb.0 as u32, _mm512_or_si512(template0, extra));
                self.splat16((to_bb.0 >> 32) as u32, _mm512_or_si512(template1, extra));
            }
        }
    }

    #[cfg(not(target_feature = "avx512vbmi2"))]
    pub fn push_pawns_setwise(&mut self, offset: i8, to_bb: Bitboard, kind: MoveKind) {
        for to in to_bb {
            self.push(to.shift(-offset), to, kind);
        }
    }

    #[cfg(target_feature = "avx512vbmi2")]
    pub fn push_pawns_setwise(&mut self, offset: i8, to_bb: Bitboard, kind: MoveKind) {
        if !to_bb.is_empty() {
            use std::{arch::x86_64::*, mem::transmute};

            unsafe {
                let template0: __m512i = transmute({
                    let mut template0: [Move; 32] = [Move::NULL; 32];
                    for (i, e) in template0.iter_mut().enumerate() {
                        let sq = Square::new(i as u8);
                        *e = Move::new(sq, sq, transmute::<u8, MoveKind>(0u8));
                    }
                    template0
                });
                let template1: __m512i = transmute({
                    let mut template1: [Move; 32] = [Move::NULL; 32];
                    for (i, e) in template1.iter_mut().enumerate() {
                        let sq = Square::new(32u8 + i as u8);
                        *e = Move::new(sq, sq, transmute::<u8, MoveKind>(0u8));
                    }
                    template1
                });

                let offset = offset as i16;
                let extra = _mm512_set1_epi16(((kind as i16) << 12).wrapping_sub(offset));

                self.splat8(to_bb.0 as u32, _mm512_add_epi16(template0, extra));
                self.splat8((to_bb.0 >> 32) as u32, _mm512_add_epi16(template1, extra));
            }
        }
    }

    pub fn push_promotion_capture_setwise(&mut self, offset: i8, to_bb: Bitboard) {
        if !to_bb.is_empty() {
            self.push_pawns_setwise(offset, to_bb, MoveKind::PromotionCaptureQ);
            self.push_pawns_setwise(offset, to_bb, MoveKind::PromotionCaptureR);
            self.push_pawns_setwise(offset, to_bb, MoveKind::PromotionCaptureB);
            self.push_pawns_setwise(offset, to_bb, MoveKind::PromotionCaptureN);
        }
    }

    pub fn moves(&self) -> &[Move] {
        unsafe { std::slice::from_raw_parts(self.moves.as_ptr().cast(), self.len) }
    }

    pub fn scores(&self) -> &[i32] {
        unsafe { std::slice::from_raw_parts(self.scores.as_ptr().cast(), self.len) }
    }

    pub fn moves_and_scores_mut(&mut self) -> (&[Move], &mut [i32]) {
        unsafe {
            (
                std::slice::from_raw_parts(self.moves.as_ptr().cast(), self.len),
                std::slice::from_raw_parts_mut(self.scores.as_mut_ptr().cast(), self.len),
            )
        }
    }

    pub fn remove(&mut self, index: usize) -> MoveEntry {
        let mv = unsafe { self.moves[index].assume_init() };
        let score = unsafe { self.scores[index].assume_init() };
        self.len -= 1;
        self.moves[index] = self.moves[self.len];
        self.scores[index] = self.scores[self.len];
        MoveEntry { mv, score }
    }

    #[cfg(target_feature = "avx512vbmi2")]
    unsafe fn splat8(&mut self, mask: u32, vector: std::arch::x86_64::__m512i) {
        use std::arch::x86_64::*;
        let count = mask.count_ones() as usize;
        let compressed = _mm512_maskz_compress_epi16(mask, vector);
        _mm_storeu_si128(
            self.moves[self.len..].as_mut_ptr().cast(),
            _mm512_castsi512_si128(compressed),
        );
        self.len += count;
    }

    #[cfg(target_feature = "avx512vbmi2")]
    unsafe fn splat16(&mut self, mask: u32, vector: std::arch::x86_64::__m512i) {
        use std::arch::x86_64::*;
        let count = mask.count_ones() as usize;
        let compressed = _mm512_maskz_compress_epi16(mask, vector);
        _mm256_storeu_si256(
            self.moves[self.len..].as_mut_ptr().cast(),
            _mm512_castsi512_si256(compressed),
        );
        self.len += count;
    }
}

impl Default for MoveList {
    fn default() -> Self {
        Self::new()
    }
}
