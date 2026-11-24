//! The direct lowering (`lower`) written into a DXIL module through `gpu::dxil`'s builder:
//! the per-lane form of a program as a signed container, with no HLSL and no DXC.
//!
//! `Ir` maps one to one onto the builder: blocks, phis (`phi_at` / `add_incoming`), integer
//! ops, compares, `select`, the thread / group / lane ids, `waveActiveBallot`, and raw buffer
//! loads and stores on the ABI's root UAVs (u0 `io`, u1 `res`, u2 `mem`). A phi the SSA
//! construction finds trivial has its uses replaced (`Module::replace_all_uses`); the phi itself
//! stays in its block, unused.

use super::lower::{self, Bin, Cmp, Ir};
use super::ssa::{Sink, Var};
use crate::gpu::dxil::container::Stage;
use crate::gpu::dxil::module::{BinOp, Block, CastOp, Pred, RmwOp, Value};
use crate::gpu::dxil::ops::{barrier, GroupShared};
use crate::gpu::dxil::shader::{RawUav, Shader};
use crate::gpu::rdna2::Decoded;

struct DxilIr {
    s: Shader,
    io: RawUav,
    res: Option<RawUav>,
    mem: RawUav,
    blocks: Vec<Block>,
    /// a wave64 program in two hardware wave32 halves (`docs/wave64-on-wave32.md`)
    split: Option<Split>,
    split_threads: Option<u32>,
}

/// The split form's state, set up in the entry block: the console thread each hardware thread
/// stands for (`xcl_x_thread`'s arrival order) and the halves' exchange words.
#[derive(Clone, Copy)]
struct Split {
    xs: GroupShared,
    /// console thread in the group, lane (0..63), half (0 / 1), console wave in the group
    tid: Value,
    lane: Value,
    half: Value,
    cw: Value,
    /// exchanges so far (an alloca: no SSA of its own runs here)
    xk: Value,
}

impl DxilIr {
    fn new(kernel: bool, split_threads: Option<u32>) -> DxilIr {
        let mut s = Shader::new(Stage::Compute, (6, 6));
        let io = s.raw_uav(0, 0);
        let res = kernel.then(|| s.raw_uav(1, 0));
        // compute: other waves may write the memory this one reads (as the HLSL path declares it)
        let mem = if kernel { s.raw_uav(2, 0) } else { s.raw_uav_coherent(2, 0) };
        DxilIr { s, io, res, mem, blocks: Vec::new(), split: None, split_threads }
    }

    fn ty(&mut self, var: Var) -> crate::gpu::dxil::module::Ty {
        match var {
            Var::M(_) | Var::Scc => self.s.m.i1(),
            Var::S(_) | Var::V(_) => self.s.m.i32(),
        }
    }

    fn bytes(&mut self, dword: Value) -> Value {
        let four = self.s.m.i32c(4);
        self.s.m.bin(BinOp::Mul, dword, four)
    }

    fn c(&mut self, v: u32) -> Value {
        self.s.m.i32c(v)
    }

    fn bin2(&mut self, op: BinOp, a: Value, b: u32) -> Value {
        let k = self.c(b);
        self.s.m.bin(op, a, k)
    }

    /// The split form's entry: the lane counter zeroed, then each hardware wave takes the next
    /// 32 console threads (`setup_lanes_compute`'s split branch).
    fn setup_split(&mut self, total: u32) {
        let waves = total.div_ceil(64).max(1);
        let i32 = self.s.m.i32();
        let xs = self.s.groupshared("xcl_xs", i32, 4 * waves as u64);
        let xw = self.s.groupshared("xcl_xw", i32, 1);
        let xk = self.s.m.alloca(i32, 4);
        let z = self.c(0);
        self.s.m.store(xk, z, 4);
        let p = self.s.gs_ptr(xw, z);
        self.s.m.store(p, z, 4);
        self.s.barrier(barrier::GROUP_SYNC);
        let t = self.s.m.i1c(true);
        let n = self.s.wave_all_bit_count(t);
        let k32 = self.c(32);
        let full = self.s.m.cmp(Pred::Eq, n, k32);
        let first = self.s.wave_is_first_lane();
        let take = self.s.m.bin(BinOp::And, first, full);
        let k = self.s.m.select(take, k32, z);
        let got = self.s.m.atomic_rmw(RmwOp::Add, p, k);
        let dflt = self.c(total & !31);
        let base = self.s.m.select(take, got, dflt);
        let base = self.s.wave_read_lane_first(base);
        let t = self.s.m.i1c(true);
        let rank = self.s.wave_prefix_bit_count(t);
        let tid = self.s.m.bin(BinOp::Add, base, rank);
        let lane = self.bin2(BinOp::And, tid, 63);
        let half = self.bin2(BinOp::LShr, lane, 5);
        let cw = self.bin2(BinOp::LShr, tid, 6);
        self.split = Some(Split { xs, tid, lane, half, cw, xk });
    }

    /// The slot of `half` for the exchange of parity `par` in this console wave's words.
    fn slot(&mut self, sp: Split, half: Value, par: Value) -> Value {
        let x = self.bin2(BinOp::Shl, sp.cw, 1);
        let x = self.s.m.bin(BinOp::Add, x, half);
        let x = self.bin2(BinOp::Shl, x, 1);
        self.s.m.bin(BinOp::Add, x, par)
    }

    /// The split form's exchange: this half gives `v` (the same in all its lanes), gets the
    /// other half's. The halves are the whole group (one console wave of 33–64 threads, as
    /// `gpu::dxil::lanes`'s `Partner::Present`): they meet at a group barrier, which needs no
    /// block of its own, so the lowering's block order (its dominance order) stays as it is.
    /// The lowering refuses a whole-wave operation where some lanes may be elsewhere, so both
    /// halves reach the same exchanges in the same order and every barrier is reached by the
    /// whole group. A slot per (half, parity of the exchange count): the partner can be at most
    /// one exchange ahead, so it never overwrites the word being read.
    fn exchange(&mut self, v: Value) -> Value {
        let sp = self.split.expect("split form");
        if self.split_threads.is_some_and(|t| t <= 32) {
            // a console wave of at most 32 threads: no partner half, its lanes read as 0
            return self.c(0);
        }
        let k = self.s.m.load(sp.xk, 4);
        let k1 = self.bin2(BinOp::Add, k, 1);
        self.s.m.store(sp.xk, k1, 4);
        let par = self.bin2(BinOp::And, k, 1);
        let mine = self.slot(sp, sp.half, par);
        let p = self.s.gs_ptr(sp.xs, mine);
        self.s.m.store(p, v, 4);
        self.s.barrier(barrier::GROUP_SYNC);
        let ho = self.bin2(BinOp::Xor, sp.half, 1);
        let theirs = self.slot(sp, ho, par);
        let q = self.s.gs_ptr(sp.xs, theirs);
        self.s.m.load(q, 4)
    }
}

impl Sink for DxilIr {
    type V = Value;
    fn phi(&mut self, block: usize, var: Var) -> Value {
        let t = self.ty(var);
        self.s.m.phi_at(self.blocks[block], t)
    }
    fn incoming(&mut self, phi: Value, value: Value, pred: usize) {
        self.s.m.add_incoming(phi, value, self.blocks[pred]);
    }
    fn replace(&mut self, old: Value, new: Value) {
        // every use of the trivial phi becomes `new`; the phi stays in its block, unused
        self.s.m.replace_all_uses(old, new);
    }
    fn undef(&mut self, var: Var) -> Value {
        let t = self.ty(var);
        self.s.m.undef(t)
    }
}

impl Ir for DxilIr {
    fn new_block(&mut self) -> usize {
        let first = self.blocks.is_empty();
        let b = if first { self.s.m.entry_block() } else { self.s.m.new_block() };
        self.blocks.push(b);
        if first {
            if let Some(t) = self.split_threads {
                self.s.m.set_block(b);
                self.setup_split(t);
            }
        }
        self.blocks.len() - 1
    }
    fn at(&mut self, block: usize) {
        self.s.m.set_block(self.blocks[block]);
    }
    fn int(&mut self, v: u32) -> Value {
        self.s.m.i32c(v)
    }
    fn boolean(&mut self, v: bool) -> Value {
        self.s.m.i1c(v)
    }
    fn bin(&mut self, op: Bin, a: Value, b: Value) -> Value {
        let op = match op {
            Bin::Add => BinOp::Add,
            Bin::Sub => BinOp::Sub,
            Bin::Mul => BinOp::Mul,
            Bin::And => BinOp::And,
            Bin::Or => BinOp::Or,
            Bin::Xor => BinOp::Xor,
            Bin::Shl => BinOp::Shl,
            Bin::LShr => BinOp::LShr,
            Bin::AShr => BinOp::AShr,
        };
        self.s.m.bin(op, a, b)
    }
    fn icmp(&mut self, c: Cmp, a: Value, b: Value) -> Value {
        let p = match c {
            Cmp::Eq => Pred::Eq,
            Cmp::Ne => Pred::Ne,
            Cmp::Ult => Pred::ULt,
            Cmp::Ule => Pred::ULe,
            Cmp::Ugt => Pred::UGt,
            Cmp::Uge => Pred::UGe,
            Cmp::Slt => Pred::SLt,
            Cmp::Sle => Pred::SLe,
            Cmp::Sgt => Pred::SGt,
            Cmp::Sge => Pred::SGe,
        };
        self.s.m.cmp(p, a, b)
    }
    fn select(&mut self, c: Value, a: Value, b: Value) -> Value {
        self.s.m.select(c, a, b)
    }
    fn zext(&mut self, b: Value) -> Value {
        let t = self.s.m.i32();
        self.s.m.cast(CastOp::ZExt, b, t)
    }
    fn lane(&mut self) -> Value {
        match self.split {
            Some(sp) => sp.lane,
            None => self.s.wave_lane_index(),
        }
    }
    fn whole_needs_exchange(&self) -> bool {
        self.split.is_some()
    }
    fn group_id(&mut self, k: u32) -> Value {
        self.s.group_id(k)
    }
    fn compute_ids(&mut self, threads: [u32; 3]) -> (Value, [Value; 3]) {
        // 1-D groups only until `Lw::setup_lanes_compute` numbers 2-D / 3-D threads by arrival
        // (`compute` refuses the others): a 1-D group's waves are its consecutive threads; in
        // the split form, the console thread the arrival order gave this one
        debug_assert!(threads[1] * threads[2] == 1);
        let flat = match self.split {
            Some(sp) => sp.tid,
            None => self.s.thread_id_in_group(0),
        };
        let zero = self.s.m.i32c(0);
        (flat, [flat, zero, zero])
    }
    fn ballot(&mut self, c: Value) -> [Value; 2] {
        let w = self.s.wave_ballot(c);
        let Some(sp) = self.split else { return [w[0], w[1]] };
        // this half's word, and the other half's through an exchange
        let other = self.exchange(w[0]);
        let z = self.c(0);
        let low = self.s.m.cmp(Pred::Eq, sp.half, z);
        let lo = self.s.m.select(low, w[0], other);
        let hi = self.s.m.select(low, other, w[0]);
        [lo, hi]
    }
    // one hardware wave per console wave (native wave64, or wave32): the half-local forms are
    // the same operations; split: this half's word in both places (only its own is right)
    fn ballot_half(&mut self, c: Value) -> [Value; 2] {
        let w = self.s.wave_ballot(c);
        [w[0], if self.split.is_some() { w[0] } else { w[1] }]
    }
    fn read_lane(&mut self, v: Value, lane: Value) -> Value {
        let Some(sp) = self.split else { return self.s.wave_read_lane_at(v, lane) };
        // `lane` is the same in every lane (readlane, readfirstlane): each half reads it in its
        // own 32 and gives that; the owner half's value is the answer
        let l31 = self.bin2(BinOp::And, lane, 31);
        let m = self.s.wave_read_lane_at(v, l31);
        let o = self.exchange(m);
        let lh = self.bin2(BinOp::LShr, lane, 5);
        let same = self.s.m.cmp(Pred::Eq, lh, sp.half);
        self.s.m.select(same, m, o)
    }
    fn read_lane_half(&mut self, v: Value, lane: Value) -> Value {
        if self.split.is_some() {
            let l31 = self.bin2(BinOp::And, lane, 31);
            return self.s.wave_read_lane_at(v, l31);
        }
        self.s.wave_read_lane_at(v, lane)
    }
    fn count_bits(&mut self, v: Value) -> Value {
        self.s.bit_count(crate::gpu::dxil::ops::BitCount::Countbits, v)
    }
    fn io_load(&mut self, dword: Value) -> Value {
        let a = self.bytes(dword);
        let io = self.io;
        self.s.raw_load(io, a, 1)[0]
    }
    fn res_store(&mut self, dword: Value, value: Value) {
        let a = self.bytes(dword);
        let res = self.res.expect("res is bound in kernel mode");
        self.s.raw_store(res, a, &[value]);
    }
    fn mem_load(&mut self, addr: Value) -> Value {
        let mem = self.mem;
        self.s.raw_load(mem, addr, 1)[0]
    }
    fn mem_store(&mut self, addr: Value, value: Value) {
        let mem = self.mem;
        self.s.raw_store(mem, addr, &[value]);
    }
    fn br(&mut self, target: usize) {
        self.s.m.br(self.blocks[target]);
    }
    fn br_if(&mut self, c: Value, then: usize, otherwise: usize) {
        self.s.m.cond_br(c, self.blocks[then], self.blocks[otherwise]);
    }
    fn ret(&mut self) {
        self.s.m.ret();
    }
}

/// A kernel-mode program in its per-lane form, as a signed DXIL container (one wave of
/// `WaveSize` = the program's wave size), and the masks whose final value may differ from
/// the console's (`lower::lower_kernel`).
pub fn kernel(insts: &[Decoded], wave64: bool) -> Result<(Vec<u8>, Vec<u8>), String> {
    let (ir, stale) = lower::lower_kernel(insts, wave64, DxilIr::new(true, None))?;
    let lanes = if wave64 { 64 } else { 32 };
    Ok((ir.s.finish([lanes, 1, 1], Some(lanes)), stale))
}

/// `compute` from the program's code (decoded, far jumps resolved).
pub fn compute_code(code: &[u8], cs: &super::CsInterface, wave64: bool) -> Result<Vec<u8>, String> {
    let mut insts = Vec::new();
    let mut off = 0usize;
    while off + 4 <= code.len() {
        let d = crate::gpu::rdna2::decode(&code[off..], off as u64).map_err(|e| format!("cannot decode ({e:?})"))?;
        off += d.len;
        insts.push(d);
    }
    crate::gpu::rdna2::resolve_far_jumps(&mut insts);
    compute(&insts, cs, wave64)
}

/// Survey: every reason the per-lane form refuses a compute program (`lower::survey_compute`).
pub fn survey_compute_code(code: &[u8], cs: &super::CsInterface, wave64: bool) -> Vec<String> {
    let mut insts = Vec::new();
    let mut off = 0usize;
    while off + 4 <= code.len() {
        match crate::gpu::rdna2::decode(&code[off..], off as u64) {
            Ok(d) => {
                off += d.len;
                insts.push(d);
            }
            Err(e) => return vec![format!("cannot decode ({e:?})")],
        }
    }
    crate::gpu::rdna2::resolve_far_jumps(&mut insts);
    // a 2-D / 3-D group is noted as `compute` refuses it, and the rest surveyed as a 1-D group
    // of as many threads (the thread ids differ, the instructions are the same)
    if cs.threads[1] * cs.threads[2] > 1 {
        let flat = super::CsInterface { threads: [cs.threads.iter().product(), 1, 1], ..cs.clone() };
        let mut why = vec!["2-D / 3-D thread group (waits for Lw::setup_lanes_compute)".to_string()];
        why.extend(lower::survey_compute(&insts, &flat, wave64, DxilIr::new(false, None)));
        return why;
    }
    lower::survey_compute(&insts, cs, wave64, DxilIr::new(false, None))
}

/// A compute program in its per-lane form, as a signed DXIL container.
pub fn compute(insts: &[Decoded], cs: &super::CsInterface, wave64: bool) -> Result<Vec<u8>, String> {
    if cs.threads[1] * cs.threads[2] > 1 {
        return Err("2-D / 3-D thread group (waits for Lw::setup_lanes_compute)".into());
    }
    let ir = lower::lower_compute(insts, cs, wave64, DxilIr::new(false, None))?;
    Ok(ir.s.finish(cs.threads, Some(if wave64 { 64 } else { 32 })))
}

/// `kernel` of a wave64 program for a GPU with 32-lane waves only: the split form, two
/// hardware wave32s per console wave exchanging through group-shared memory.
/// EXEC at launch must be full: a lane outside it goes straight to the exit and would miss the
/// barriers the halves meet at in the program (compute waves always start with every lane).
pub fn kernel_split(insts: &[Decoded]) -> Result<(Vec<u8>, Vec<u8>), String> {
    let (ir, stale) = lower::lower_kernel(insts, true, DxilIr::new(true, Some(64)))?;
    Ok((ir.s.finish([64, 1, 1], Some(32)), stale))
}

/// `compute` of a wave64 program in the split form (1-D groups).
pub fn compute_split(insts: &[Decoded], cs: &super::CsInterface) -> Result<Vec<u8>, String> {
    if cs.threads[1] * cs.threads[2] != 1 {
        return Err("split form: 2-D / 3-D groups stay on HLSL".into());
    }
    if cs.threads[0] > 64 {
        // several console waves per group: their halves would pair up through arrival
        // counters and a polled spin, whose loop needs blocks of its own (not yet here)
        return Err("split form: groups of several console waves stay on HLSL".into());
    }
    let ir = lower::lower_compute(insts, cs, true, DxilIr::new(false, Some(cs.threads[0])))?;
    Ok(ir.s.finish(cs.threads, Some(32)))
}

#[cfg(test)]
mod tests {
    use super::super::lower::tests::*;
    use super::super::{dxc, runner, translate, Mode, Options};

    fn gpu() -> bool {
        if runner::adapter().is_err() {
            runner::skip_test("no GPU");
            return false;
        }
        true
    }

    /// The per-lane form as real DXIL on the GPU agrees with the lowering run on the CPU (and so
    /// with the HLSL path, which `lower`'s tests compare it with): registers and memory.
    #[test]
    fn per_lane_kernels_run_on_the_gpu() {
        if !gpu() {
            return;
        }
        for (name, lines) in [("straight", STRAIGHT), ("skipped if", SKIPPED_IF), ("leaving loop", LEAVING_LOOP), ("uniform loop", UNIFORM_LOOP), ("memory", MEMORY)] {
            let insts = decode(lines);
            let (dxil, stale) = super::kernel(&insts, false).unwrap_or_else(|e| panic!("{name}: {e}"));
            let (s, v) = state(32);
            let mut st = runner::KernelState::default();
            st.sgpr.copy_from_slice(&s);
            for (l, regs) in v.iter().enumerate() {
                st.vgpr[l][..regs.len()].copy_from_slice(regs);
            }
            let mem = memory_image();
            let gpu = runner::run(&dxil, &st, &mem).unwrap_or_else(|e| panic!("{name}: {e}"));
            let (cs, cv, cmem) = run_mem(lines, false, &s, &v, &mem);
            assert_eq!(&gpu.mem[..cmem.len()], &cmem[..], "{name}: memory");
            let mut sregs: Vec<usize> = vec![106, 126];
            let mut vregs: Vec<usize> = Vec::new();
            for d in &insts {
                let (r, w) = crate::gpu::rdna2::sgpr_uses(d);
                sregs.extend(r.iter().chain(&w).map(|&x| x as usize).filter(|&x| x < 125));
                let (r, w) = crate::gpu::rdna2::vgpr_uses(d);
                vregs.extend(r.iter().chain(&w).map(|&x| x as usize));
            }
            for &r in sregs.iter().filter(|&&r| !stale.contains(&(r as u8))) {
                assert_eq!(gpu.sgpr[r], cs[r], "{name}: s{r}");
            }
            for l in 0..32 {
                for &r in &vregs {
                    assert_eq!(gpu.vgpr[l][r], cv[l][r], "{name}: lane {l} v{r}");
                }
            }
        }
    }

    #[test]
    fn a_per_lane_compute_program_runs_on_the_gpu() {
        if !gpu() {
            return;
        }
        let cs = compute_cs();
        let dxil = super::compute(&decode(COMPUTE), &cs, false).unwrap();
        let mem = vec![0u32; 256];
        let gpu = runner::dispatch(&dxil, [3, 1, 1], &[100, 0], &mem).unwrap();
        let cpu = run_compute(COMPUTE, &cs, 3, &[100, 0], &mem);
        assert_eq!(&gpu[..cpu.len()], &cpu[..]);
        // and the HLSL path, for good measure
        if dxc::location().is_ok() {
            let bytes: Vec<u8> = COMPUTE.iter().flat_map(|(n, e, f)| crate::gpu::rdna2::encode(n, e, f, None).unwrap()).collect();
            let t = translate(&bytes, &Options { mode: Mode::Compute(cs.clone()), wave64: false, ..Default::default() }).unwrap();
            let hlsl = dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).unwrap();
            assert_eq!(runner::dispatch(&hlsl, [3, 1, 1], &[100, 0], &mem).unwrap(), gpu);
        }
    }
}
