# great-wall-core

**The Wall.** Fractal encoder engine — bit-exact encoding and decoding of seed entropy onto a user-specific perturbation of the Burning Ship fractal.

Part of [Great Wall](https://github.com/Yuri-SVB/Great-Wallet), which pins
this library to a compatible commit. This repository has no submodules.

**Status:** Beta (public).

**Documentation:** [great-wall-docs/great-wall-core](https://github.com/Yuri-SVB/great-wall-docs/tree/main/great-wall-core)
— design, architecture and threat model live there, not here.

## Build

```sh
cargo build --release      # engine
cargo test                 # Rust + bijection + frozen vectors
python3 viewer.py          # reference viewer
```

**⚡ Support:** [github.com/Yuri-SVB/SUPPORT](https://github.com/Yuri-SVB/SUPPORT) — free software, nothing gated.

## License

MIT or Apache-2.0, at your option.
