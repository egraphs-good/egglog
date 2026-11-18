use egglog_numeric_id::{NumericId, define_id};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

// Preserve scalar JSON encodings while giving each numeric namespace a type.
macro_rules! serial_id {
    ($name:ident, $repr:tt, $doc:literal) => {
        define_id!(pub $name, $repr, $doc);
        impl Serialize for $name {
            fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
                self.rep().serialize(serializer)
            }
        }
        impl<'de> Deserialize<'de> for $name {
            fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
                <$repr>::deserialize(deserializer).map(Self::new)
            }
        }
    };
}

serial_id!(
    FormatVersion,
    u32,
    "Version of the surrounding JSON format, not an e-class or partition ID. Version 1 is currently supported."
);
impl FormatVersion {
    pub const V1: Self = Self::new_const(1);
}
define_id!(pub(crate) ClassId, usize, "Dense e-class index in a comparison graph. In joint refinement, right-side IDs follow all left-side IDs. Unrelated to serialized class names.");
serial_id!(
    BlockId,
    usize,
    "Equivalence block in one refinement partition. IDs are local to that partition and round, not stable e-class identities."
);
define_id!(pub RowId, usize, "Index in one input database's rows, used by the representative worklist.");
define_id!(pub(crate) SymbolId, usize, "Interned complete node label, shared by both inputs during joint refinement.");
