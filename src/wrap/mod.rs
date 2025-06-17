pub use derive_more;
pub mod tx;
pub mod tx_rx_vt;
pub mod tx_vt;
mod wrap;
pub use wrap::*;
pub mod tx_minimal;

/// macro to quickly define a Transimitter with no version control
#[macro_export]
macro_rules! basic_tx_no_vt {
    ($name:ident) => {
        struct $name {
            tx: egglog::wrap::tx::TxNoVT,
        }
        impl SingletonGetter for $name {
            type RetTy = egglog::wrap::tx::TxNoVT;
            fn sgl() -> &'static egglog::wrap::tx::TxNoVT {
                static INSTANCE: std::sync::OnceLock<$name> = std::sync::OnceLock::new();
                &INSTANCE
                    .get_or_init(|| -> $name {
                        Self {
                            tx: egglog::wrap::tx::TxNoVT::new(),
                        }
                    })
                    .tx
            }
        }
    };
}
/// macro to quickly define a Transimitter with version control
#[macro_export]
macro_rules! basic_tx_vt {
    ($name:ident) => {
        struct $name {
            tx: egglog::wrap::tx_vt::TxVT,
        }
        impl SingletonGetter for $name {
            type RetTy = egglog::wrap::tx_vt::TxVT;
            fn sgl() -> &'static egglog::wrap::tx_vt::TxVT {
                static INSTANCE: std::sync::OnceLock<$name> = std::sync::OnceLock::new();
                &INSTANCE
                    .get_or_init(|| -> $name {
                        Self {
                            tx: egglog::wrap::tx_vt::TxVT::new(),
                        }
                    })
                    .tx
            }
        }
    };
}
/// macro to quickly define a minimal Transimitter
#[macro_export]
macro_rules! basic_tx_minimal {
    ($name:ident) => {
        struct $name {
            tx: egglog::wrap::tx_minimal::TxMinimal,
        }
        impl SingletonGetter for $name {
            type RetTy = egglog::wrap::tx_minimal::TxMinimal;
            fn sgl() -> &'static egglog::wrap::tx_minimal::TxMinimal {
                static INSTANCE: std::sync::OnceLock<$name> = std::sync::OnceLock::new();
                &INSTANCE
                    .get_or_init(|| -> $name {
                        Self {
                            tx: egglog::wrap::tx_minimal::TxMinimal::new(),
                        }
                    })
                    .tx
            }
        }
    };
}

#[macro_export]
macro_rules! basic_tx_rx_vt {
    ($name:ident) => {
        struct $name {
            tx: egglog::wrap::tx_rx_vt::TxRxVT,
        }
        impl SingletonGetter for $name {
            type RetTy = egglog::wrap::tx_rx_vt::TxRxVT;
            fn sgl() -> &'static egglog::wrap::tx_rx_vt::TxRxVT {
                static INSTANCE: std::sync::OnceLock<$name> = std::sync::OnceLock::new();
                &INSTANCE
                    .get_or_init(|| -> $name {
                        Self {
                            tx: egglog::wrap::tx_rx_vt::TxRxVT::new(),
                        }
                    })
                    .tx
            }
        }
    };
}
