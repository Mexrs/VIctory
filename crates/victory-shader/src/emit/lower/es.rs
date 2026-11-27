//! Export-stage lowering: the merged export program's prologue masks
//! and ring writes, moved out of `lower.rs` so that file stays under
//! the asset policy's 512 KiB limit.

use std::collections::HashSet;

use super::{Lower, Why};
use crate::emit::bitcode::{Attr, V};
use crate::gfx10::{Fields, Operand};
use crate::io::LdsOutput;
use crate::ir::{EXEC, Program, Reg, Value};

/// The lane masks (and the SCC beside them) a merged export or geometry
/// program's prologue makes from its lane count in s3: `s_bfe_{i32,u32}
/// count, s3, ...; s_bfm_b64 exec, count, 0; s_cmp_eq_u32 count, 64;
/// s_cselect_b64 exec, -1, exec`.
pub(crate) fn merged_wave_masks(p: &Program) -> HashSet<Value> {
    let mut out = HashSet::new();
    let Some(s3) = p.inputs.iter().find(|i| i.0 == Reg::S(3)).map(|i| i.1) else {
        return out;
    };
    let Some(entry) = p.blocks.first() else {
        return out;
    };
    let mut counts = HashSet::new();
    let mut bfm = HashSet::new();
    for inst in &entry.insts {
        let Fields::Sop { dst, src, .. } = &inst.fields else {
            continue;
        };
        match inst.name() {
            "s_bfe_i32" | "s_bfe_u32" => {
                if matches!(src.first(), Some(Operand::Scalar { code: 3, .. }))
                    && inst.read(Reg::S(3)) == Some(s3)
                    && let Some(&Operand::Scalar { code, .. }) = dst.as_ref()
                    && let Some(w) = inst.write(Reg::S(code))
                {
                    counts.insert(w);
                }
            }
            "s_bfm_b64" => {
                if matches!(dst, Some(Operand::Scalar { code, .. }) if *code == EXEC)
                    && matches!(src.first(), Some(Operand::Scalar { code, .. }) if inst.read(Reg::S(*code)).is_some_and(|v| counts.contains(&v)))
                    && matches!(src.get(1), Some(Operand::Const(128)))
                    && let Some(w) = inst.write(Reg::S(EXEC))
                {
                    bfm.insert(w);
                    out.insert(w);
                }
            }
            "s_cmp_eq_u32" => {
                if matches!(src.first(), Some(Operand::Scalar { code, .. }) if inst.read(Reg::S(*code)).is_some_and(|v| counts.contains(&v)))
                    && matches!(
                        src.get(1),
                        Some(Operand::Const(192)) | Some(Operand::Literal(64))
                    )
                    && let Some(w) = inst.write(Reg::Scc)
                {
                    out.insert(w);
                }
            }
            "s_cselect_b64" => {
                if matches!(dst, Some(Operand::Scalar { code, .. }) if *code == EXEC)
                    && matches!(src.first(), Some(Operand::Const(193)))
                    && matches!(src.get(1), Some(Operand::Scalar { code, .. }) if *code == EXEC)
                    && inst.read(Reg::S(EXEC)).is_some_and(|v| bfm.contains(&v))
                    && let Some(w) = inst.write(Reg::S(EXEC))
                {
                    out.insert(w);
                }
            }
            _ => {}
        }
    }
    out
}

impl<'a> Lower<'a> {
    /// An export program's ring writes, as the vertex shader's outputs.
    pub(crate) fn store_lds_outputs(&mut self, writes: &[LdsOutput], guard: Option<V>) -> Result<(), Why> {
        // Dwords of rows the signature lacks are padding.
        let mut stores = Vec::new();
        for out in writes {
            let found = self.s.outputs.iter().position(|o| {
                o.row == Some(u32::from(out.row))
                    && out.channel >= o.col
                    && out.channel < o.col + o.cols
            });
            let Some(id) = found else { continue };
            if self.s.outputs[id].half {
                // shortcut: 16-bit outputs (as `export` packs them) are
                // not followed into LDS.
                return Err("a 16-bit output of a merged geometry stage");
            }
            let x = self.i32_of(out.value)?;
            stores.push((id, out.channel - self.s.outputs[id].col, out.value, x));
        }
        let at = self.begin_guarded(guard);
        for (id, col, value, x) in stores {
            let el = &mut self.s.outputs[id];
            el.usage |= 1 << col;
            let (suffix, x) = match el.comp {
                9 => {
                    let f = match self.floats.get(&value) {
                        Some(&f) => f,
                        None => self.as_float(x),
                    };
                    (".f32", f)
                }
                _ => (".i32", x),
            };
            let (id, zero) = (self.s.m.i32(id as u32), self.s.m.i32(0));
            let args = [id, zero, self.s.m.i8(col), x];
            let void = self.s.t.void;
            let name = format!("storeOutput{suffix}");
            self.s.op(self.cur, &name, Attr::NoUnwind, void, 5, &args);
        }
        self.end_guarded(at, &[]);
        Ok(())
    }

}
