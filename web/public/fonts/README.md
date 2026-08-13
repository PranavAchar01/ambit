# Computer Modern webfonts

CMU Serif and CMU Typewriter Text, self-hosted so the UI needs no CDN and works offline.

Source: the `computer-modern` npm package (v0.1.3), which repackages the Computer Modern Unicode
fonts. Licensed under the SIL Open Font License — see `OFL.txt`.

Self-hosted rather than imported from the package because that package declares
`font-style: roman`, which is not a valid CSS value; browsers may discard the whole `@font-face`
rule. `app/globals.css` re-declares these files with `font-style: normal`.
