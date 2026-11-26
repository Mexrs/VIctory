//! Geometry-stage lowering: the single-stream triangle geometry
//! programs of decision 0003's lowering, moved out of `lower.rs` so
//! that file stays under the asset policy's 512 KiB limit.

use super::{Lower, StageIo, Stop, Val, Why, unsupported};
use crate::emit::bitcode::{Attr, Cast};
use crate::ir::Inst;

impl<'a> Lower<'a> {
    fn load_vertex_input(&mut self, id: usize, col: u8, vertex: Option<u8>) -> Val {
        let e = &mut self.s.inputs[id];
        e.usage |= 1 << col;
        let (int, bit) = (e.comp != 9, e.comp == 1);
        let ty = if bit {
            self.s.t.i1
        } else if int {
            self.s.t.i32
        } else {
            self.s.t.f32
        };
        let suffix = if bit {
            ".i1"
        } else if int {
            ".i32"
        } else {
            ".f32"
        };
        let entry = self.s.entry;
        let index = match vertex {
            Some(n) => self.s.m.i32(u32::from(n)),
            None => self.s.m.undef(self.s.t.i32),
        };
        let args = [
            self.s.m.i32(id as u32),
            self.s.m.i32(0),
            self.s.m.i8(col),
            index,
        ];
        let v = self.s.op(
            entry,
            &format!("loadInput{suffix}"),
            Attr::ReadNone,
            ty,
            4,
            &args,
        );
        if bit {
            let z = self.s.m.cast(entry, Cast::ZExt, v, self.s.t.i32);
            Val::I32(z)
        } else if int {
            Val::I32(v)
        } else {
            Val::I32(self.s.m.cast(entry, Cast::Bitcast, v, self.s.t.i32))
        }
    }

    /// A geometry program's instruction: ring addresses, ring reads,
    /// and stream messages are handled here; anything else falls
    /// through to the ordinary instruction path. Returns whether it
    /// handled the instruction.
    fn geometry_inst(&mut self, inst: &'a Inst, offset: usize) -> Result<bool, Stop> {
        let Some(gs) = self.x.io.geometry.as_deref() else {
            return Ok(false);
        };
        if gs.address_steps.contains(&offset) {
            for &(_, value) in &inst.writes {
                self.vals.insert(
                    value,
                    Val::Missing("a geometry ring address used as shader data"),
                );
            }
            return Ok(true);
        }
        if let Some(reads) = gs.reads.get(&offset) {
            for &(v, index, word) in reads {
                let row = u8::try_from(word / 4)
                    .map_err(|_| unsupported(offset, "a geometry input row"))?;
                let (id, col) = self.find_input(row, (word % 4) as u8).ok_or(unsupported(
                    offset,
                    "a geometry input missing from the signature",
                ))?;
                if self.s.inputs[id].half || index >= 3 {
                    return Err(unsupported(offset, "a geometry input format or vertex"));
                }
                let input = self.load_vertex_input(id, col, Some(index));
                self.vals.insert(v, input);
            }
            return Ok(true);
        }
        if let Some(message) = gs.messages.get(&offset) {
            self.geometry_message(message)
                .map_err(|why| unsupported(offset, why))?;
            return Ok(true);
        }
        if inst.name() == "ds_write2_b32" {
            // Raising captured each stored SSA value at its emit site.
            return Ok(true);
        }
        Ok(false)
    }

    fn geometry_message(&mut self, message: &crate::geometry::StreamOp) -> Result<(), Why> {
        use crate::geometry::{Component, StreamOp};
        let (name, opcode) = match message {
            StreamOp::Cut => ("cutStream", 98),
            StreamOp::Emit(exports) => {
                let StageIo::Gs(gs) = &self.x.native.io else {
                    return Err("geometry stage data");
                };
                let StageIo::Vs { outputs, .. } = &gs.copy.io else {
                    return Err("geometry copy outputs");
                };
                let mut written = vec![0u8; self.s.outputs.len()];
                for export in exports {
                    let row = match export.target {
                        12 => self
                            .s
                            .outputs
                            .iter()
                            .find(|e| e.kind == 3)
                            .and_then(|e| e.row),
                        t @ 32..=63 => {
                            let o = outputs
                                .iter()
                                .find(|o| o.param == t - 32)
                                .ok_or("a geometry output parameter")?;
                            if o.fp16 || o.pack2 {
                                return Err("a packed geometry output");
                            }
                            Some(u32::from(o.row))
                        }
                        _ => return Err("a geometry copy export target"),
                    }
                    .ok_or("a geometry output signature row")?;
                    for (channel, source) in export.channels.iter().enumerate() {
                        let Some(src) = source else {
                            continue;
                        };
                        let ch = channel as u8;
                        let Some(id) = self.s.outputs.iter().position(|e| {
                            e.row == Some(row) && (e.col..e.col + e.cols).contains(&ch)
                        }) else {
                            continue;
                        };
                        if self.s.outputs[id].half {
                            return Err("a 16-bit geometry output");
                        }
                        let col = ch - self.s.outputs[id].col;
                        let bits = match *src {
                            Component::Value(v) => self.i32_of(v)?,
                            Component::Constant(n) => self.s.m.i32(n),
                        };
                        let (op, x) = if self.s.outputs[id].comp == 9 {
                            ("storeOutput.f32", self.as_float(bits))
                        } else {
                            ("storeOutput.i32", bits)
                        };
                        written[id] |= 1 << col;
                        self.s.outputs[id].usage |= 1 << col;
                        let args = [
                            self.s.m.i32(id as u32),
                            self.s.m.i32(0),
                            self.s.m.i8(col),
                            x,
                        ];
                        self.s
                            .op(self.cur, op, Attr::NoUnwind, self.s.t.void, 5, &args);
                    }
                }
                if self
                    .s
                    .outputs
                    .iter()
                    .zip(written)
                    .any(|(e, w)| w != (1 << e.cols) - 1)
                {
                    return Err("a geometry output missing from the copy exports");
                }
                ("emitStream", 97)
            }
        };
        let stream = self.s.m.i8(0);
        self.s.op(
            self.cur,
            name,
            Attr::NoUnwind,
            self.s.t.void,
            opcode,
            &[stream],
        );
        Ok(())
    }
}
