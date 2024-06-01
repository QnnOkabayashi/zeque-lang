//! Generate size and alignments for structs
use crate::thir;
use std::collections::HashMap;
use string_interner::DefaultSymbol;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("cyclic type with infinite size")]
    CyclicType,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Align(pub u32);

#[derive(Copy, Clone, Debug)]
pub struct SizeAlign {
    // in bytes
    pub size: u64,
    pub align: Align,
}

impl SizeAlign {
    const I0: Self = Self::new(0, 1);
    const I32: Self = Self::new(4, 4);

    const fn new(size: u64, align: u32) -> Self {
        assert!(align.count_ones() == 1);
        SizeAlign {
            size,
            align: Align(align),
        }
    }
}

#[derive(Clone, Debug)]
pub struct StructSizeAlign {
    // the alignment is the largest alignment
    // of all fields.
    // the size is how many bytes an instance of the struct
    // would take up when put in a contiguous array.
    // so struct { i32, i32, bool } would have
    // align 4 and size 24
    pub sizealign: SizeAlign,
    // for storing in registers/locals
    pub field_ordering: Vec<(DefaultSymbol, Align)>,
    // for storing in memory
    pub field_to_byte_offset: HashMap<DefaultSymbol, u64>,
}

impl StructSizeAlign {
    pub fn from_fields(
        fields: &[thir::StructField],
        structs: &[thir::Struct],
        repr: &mut Repr,
    ) -> Result<StructSizeAlign, Error> {
        let mut names_and_sizealigns: Vec<(DefaultSymbol, SizeAlign)> = fields
            .iter()
            .map(|field| {
                let sizealign = sizealign_of(field.ty, structs)?;
                Ok((field.name.0, sizealign))
            })
            .collect::<Result<_, Error>>()?;

        repr.order_fields(&mut names_and_sizealigns);

        // Struct alignment is the widest alignment of any field it contains.
        let struct_align = names_and_sizealigns
            .iter()
            .map(|(_, sizealign)| sizealign.align)
            .max()
            .unwrap_or(Align(1));

        fn padding(offset: u64, Align(align_to): Align) -> u64 {
            let align_to = align_to as u64;
            align_to - (offset % align_to)
        }

        let mut field_to_byte_offset = HashMap::with_capacity(names_and_sizealigns.len());
        let mut field_ordering = Vec::with_capacity(names_and_sizealigns.len());
        let mut offset: u64 = 0;

        for (name, SizeAlign { align, .. }) in names_and_sizealigns {
            offset += padding(offset, align);
            field_to_byte_offset.insert(name, offset);
            field_ordering.push((name, align));
        }
        offset += padding(offset, struct_align);

        Ok(StructSizeAlign {
            sizealign: SizeAlign {
                size: offset,
                align: struct_align,
            },
            field_ordering,
            field_to_byte_offset,
        })
    }
}

pub fn sizealign_of(ty: thir::Type, structs: &[thir::Struct]) -> Result<SizeAlign, Error> {
    match ty {
        thir::Type::Builtin(builtin) => match builtin {
            thir::Builtin::I32 => Ok(SizeAlign::I32),
            thir::Builtin::Bool => Ok(SizeAlign::I32), // for now to make things easier
        },
        thir::Type::Function(_) => Ok(SizeAlign::I0),
        thir::Type::Struct(struct_index) => structs
            .get(struct_index.index)
            .map(|struct_| struct_.struct_sizealign.sizealign)
            .ok_or(Error::CyclicType),
    }
}

pub enum Repr {
    C,
}

impl Repr {
    fn order_fields(&mut self, _fields: &mut [(DefaultSymbol, SizeAlign)]) {
        match self {
            Repr::C => {
                // don't reorder anything
            }
        }
    }
}
