# Agent Notes

## GitHub Markdown Math Rendering

- GitHub math rendering can behave poorly with `\tag{...}` in display equations. Prefer naming equations in headings or prose instead of using `\tag`.
- Avoid LaTeX spacing commands such as `\,`, `\;`, and `\!` in Markdown math because GitHub can expose them as literal punctuation during rendering.
- For KL notation, prefer `\mathrel{\Vert}` over spacing commands around `\Vert`.
- For integrals, prefer `\mathrm{d}z` over `\,dz`.
- For long display equations, use `aligned` blocks and split lines explicitly so GitHub does not collapse or wrap the formula vertically.
- Prefer `k^\ast` over `k^*` in Markdown math to avoid interaction with Markdown emphasis parsing.
