use crate::{
    lookup::king_attacks,
    search::NodeType,
    setwise::{bishop_attacks_setwise, knight_attacks_setwise, pawn_attacks_setwise, rook_attacks_setwise},
    thread::ThreadData,
    types::{ArrayVec, Bitboard, MAX_MOVES, Move, MoveEntry, MoveList, PieceType},
};

#[derive(Copy, Clone, Eq, PartialEq, PartialOrd)]
pub enum Stage {
    HashMove,
    GenerateNoisy,
    GoodNoisy,
    Quiet,
    BadNoisy,
}

pub struct MovePicker {
    list: MoveList,
    tt_move: Move,
    threshold: Option<i32>,
    stage: Stage,
    bad_noisy: ArrayVec<Move, MAX_MOVES>,
    bad_noisy_idx: usize,
}

#[cfg(target_feature = "avx512f")]
fn best_index(scores: &[i32]) -> usize {
    unsafe {
        use std::arch::x86_64::*;
        let len = scores.len();
        let ptr = scores.as_ptr();

        let load = |i: usize| _mm512_loadu_si512(ptr.add(i).cast());
        let pack = |scores: __m512i, indices: __m512i| _mm512_or_epi32(indices, _mm512_slli_epi32::<8>(scores));

        let mut i = 0;
        let mut indices = _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
        let mut best_vec = _mm512_set1_epi32(i32::MIN);

        while std::hint::black_box(i) + 16 <= len {
            let cur = pack(load(i), indices);

            best_vec = _mm512_max_epi32(best_vec, cur);

            i += 16;
            indices = _mm512_add_epi32(indices, _mm512_set1_epi32(16));
        }

        let cur = pack(load(i), indices);

        let valid = _mm512_cmplt_epi32_mask(indices, _mm512_set1_epi32(len as i32));
        best_vec = _mm512_mask_max_epi32(best_vec, valid, best_vec, cur);

        (_mm512_reduce_max_epi32(best_vec) & 0xff) as usize
    }
}

#[cfg(all(target_feature = "avx2"))]
unsafe fn _mm256_reduce_max_epi32(v: std::arch::x86_64::__m256i) -> i32 {
    use std::arch::x86_64::*;
    let hi = _mm256_extracti128_si256::<1>(v);
    let lo = _mm256_castsi256_si128(v);
    let m = _mm_max_epi32(lo, hi);
    let m = _mm_max_epi32(m, _mm_unpackhi_epi64(m, m));
    let m = _mm_max_epi32(m, _mm_srli_si128::<4>(m));
    _mm_cvtsi128_si32(m)
}

#[cfg(all(target_feature = "avx2", not(target_feature = "avx512f")))]
fn best_index(scores: &[i32]) -> usize {
    unsafe {
        use std::arch::x86_64::*;
        let len = scores.len();
        let ptr = scores.as_ptr();

        let load = |i: usize| _mm256_loadu_si256(ptr.add(i).cast());
        let pack = |s: __m256i, idx: __m256i| _mm256_or_si256(idx, _mm256_slli_epi32(s, 8));

        let mut i = 0;
        let mut indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
        let mut best_vec = _mm256_set1_epi32(i32::MIN);

        while std::hint::black_box(i) + 8 <= len {
            let cur = pack(load(i), indices);

            best_vec = _mm256_max_epi32(best_vec, cur);

            i += 8;
            indices = _mm256_add_epi32(indices, _mm256_set1_epi32(8));
        }

        let cur = pack(load(i), indices);
        let valid = _mm256_cmpgt_epi32(_mm256_set1_epi32(len as i32), indices);
        let cur_valid = _mm256_blendv_epi8(_mm256_set1_epi32(i32::MIN), cur, valid);
        best_vec = _mm256_max_epi32(best_vec, cur_valid);

        (_mm256_reduce_max_epi32(best_vec) & 0xff) as usize
    }
}

#[cfg(not(any(target_feature = "avx512f", target_feature = "avx2")))]
fn best_index(scores: &[i32]) -> usize {
    let mut best_idx = 0;
    let mut best_score = i32::MIN;
    for (i, &score) in scores.iter().enumerate() {
        if score >= best_score {
            best_idx = i;
            best_score = score;
        }
    }
    best_idx
}

impl MovePicker {
    pub const fn new(tt_move: Move) -> Self {
        Self {
            list: MoveList::new(),
            tt_move,
            threshold: None,
            stage: if tt_move.is_present() { Stage::HashMove } else { Stage::GenerateNoisy },
            bad_noisy: ArrayVec::new(),
            bad_noisy_idx: 0,
        }
    }

    pub const fn new_probcut(threshold: i32) -> Self {
        Self {
            list: MoveList::new(),
            tt_move: Move::NULL,
            threshold: Some(threshold),
            stage: Stage::GenerateNoisy,
            bad_noisy: ArrayVec::new(),
            bad_noisy_idx: 0,
        }
    }

    pub const fn new_qsearch() -> Self {
        Self {
            list: MoveList::new(),
            tt_move: Move::NULL,
            threshold: None,
            stage: Stage::GenerateNoisy,
            bad_noisy: ArrayVec::new(),
            bad_noisy_idx: 0,
        }
    }

    pub const fn stage(&self) -> Stage {
        self.stage
    }

    pub fn next<NODE: NodeType>(&mut self, td: &ThreadData, skip_quiets: bool, ply: isize) -> Option<Move> {
        if self.stage == Stage::HashMove {
            self.stage = Stage::GenerateNoisy;

            if td.board.is_legal(self.tt_move) {
                return Some(self.tt_move);
            }
        }

        if self.stage == Stage::GenerateNoisy {
            self.stage = Stage::GoodNoisy;
            td.board.append_noisy_moves(&mut self.list);
            self.score_noisy(td);
        }

        if self.stage == Stage::GoodNoisy {
            while !self.list.is_empty() {
                let entry = self.get_best_entry();
                if entry.mv == self.tt_move {
                    continue;
                }

                let threshold = self.threshold.unwrap_or_else(|| -entry.score / 45 + 111);
                if !td.board.see(entry.mv, threshold) {
                    self.bad_noisy.push(entry.mv);
                    continue;
                }

                if NODE::ROOT {
                    self.score_noisy(td);
                }

                return Some(entry.mv);
            }

            if skip_quiets {
                self.stage = Stage::BadNoisy;
            } else {
                self.stage = Stage::Quiet;
                td.board.append_quiet_moves(&mut self.list);
                self.score_quiet(td, ply);
            }
        }

        if self.stage == Stage::Quiet {
            if !skip_quiets {
                while !self.list.is_empty() {
                    let entry = self.get_best_entry();
                    if entry.mv == self.tt_move {
                        continue;
                    }

                    if NODE::ROOT {
                        self.score_quiet(td, ply);
                    }

                    return Some(entry.mv);
                }
            }

            self.stage = Stage::BadNoisy;
        }

        // Stage::BadNoisy
        if self.bad_noisy_idx < self.bad_noisy.len() {
            let mv = self.bad_noisy[self.bad_noisy_idx];
            self.bad_noisy_idx += 1;
            return Some(mv);
        }

        None
    }

    fn get_best_entry(&mut self) -> MoveEntry {
        let index = best_index(self.list.scores());
        self.list.remove(index)
    }

    fn score_noisy(&mut self, td: &ThreadData) {
        let threats = td.board.all_threats();

        let (moves, scores) = self.list.moves_and_scores_mut();
        for (&mv, score) in moves.iter().zip(scores.iter_mut()) {
            let captured = td.board.type_on(mv.capture_sq());
            let pt = td.board.type_on(mv.from());

            *score = 16 * captured.value()
                + td.noisy_history.get(threats, td.board.moved_piece(mv), mv.to(), captured)
                + 4000 * (mv.is_promotion() && mv.promo_piece_type() == PieceType::Queen) as i32
                + (200000 - 20000 * pt as i32) * td.board.in_check() as i32;
        }
    }

    fn score_quiet(&mut self, td: &ThreadData, ply: isize) {
        let threats = td.board.all_threats();
        let side = td.board.side_to_move();
        let occupancies = td.board.occupancies();
        let pawn_threats = td.board.piece_threats(PieceType::Pawn);

        let threatened = {
            let minor_threats =
                pawn_threats | td.board.piece_threats(PieceType::Knight) | td.board.piece_threats(PieceType::Bishop);
            let rook_threats = minor_threats | td.board.piece_threats(PieceType::Rook);
            [Bitboard(0), pawn_threats, pawn_threats, minor_threats, rook_threats, Bitboard(0)]
        };

        let escape = [0, 7768, 8218, 13424, 20208, 0];

        // safe squares where we can attack an opponent piece
        let offense = {
            let knight_vulnerable = (td.board.colored_pieces(!side, PieceType::Bishop) & !threats)
                | td.board.colored_pieces(!side, PieceType::Rook)
                | td.board.colored_pieces(!side, PieceType::Queen);
            let bishop_vulnerable = td.board.colored_pieces(!side, PieceType::Rook);
            let queen_orth_vulnerable = td.board.colored_pieces(!side, PieceType::Bishop) & !threats;
            let queen_diag_vulnerable = td.board.colored_pieces(!side, PieceType::Rook) & !threats;

            let p = pawn_attacks_setwise(td.board.colors(!side), !side) & !threats;
            let n = knight_attacks_setwise(knight_vulnerable) & !threats;
            let b = bishop_attacks_setwise(bishop_vulnerable, occupancies) & !threats;
            let r = Bitboard::file(td.board.king_square(!side).file()) & !threats;
            let q = (rook_attacks_setwise(queen_orth_vulnerable, occupancies)
                | bishop_attacks_setwise(queen_diag_vulnerable, occupancies))
                & !threats;

            [p, n, b, r, q, Bitboard(0)]
        };

        // don't move king wall pawns
        let my_king = td.board.king_square(side);
        let wall_pawns = if Bitboard::HOME_ROWS[side].contains(my_king) {
            king_attacks(my_king) & td.board.pieces(PieceType::Pawn)
        } else {
            Bitboard(0)
        };

        let (moves, scores) = self.list.moves_and_scores_mut();
        for (&mv, score) in moves.iter().zip(scores.iter_mut()) {
            let pt = td.board.type_on(mv.from());

            *score = 2048 * td.quiet_history.get(threats, side, mv) / 1024
                + 1536 * td.conthist(ply, 1, mv) / 1024
                + td.conthist(ply, 2, mv)
                + td.conthist(ply, 4, mv)
                + td.conthist(ply, 6, mv)
                + escape[pt] * threatened[pt].contains(mv.from()) as i32
                + 9325 * td.board.checking_squares(pt).contains(mv.to()) as i32
                - 7584 * threatened[pt].contains(mv.to()) as i32
                + 5000 * offense[pt].contains(mv.to()) as i32
                - 4000 * wall_pawns.contains(mv.from()) as i32;
        }
    }
}
