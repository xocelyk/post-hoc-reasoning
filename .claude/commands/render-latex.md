# render-latex

This command renders a LaTeX file as a PDF and opens it in the default PDF viewer.

## Usage

When working on a LaTeX file or when the user asks to render/compile a LaTeX document, use the render_latex.sh script located at `/Users/kyle/Documents/ws/post-hoc-reasoning/render_latex.sh`.

## Example

```bash
/Users/kyle/Documents/ws/post-hoc-reasoning/render_latex.sh /path/to/file.tex
```

## Common use cases

1. **Rendering the main paper**: 
   ```bash
   /Users/kyle/Documents/ws/post-hoc-reasoning/render_latex.sh /Users/kyle/Documents/ws/post-hoc-reasoning/overleaf/main.tex
   ```

2. **Rendering nips-draft.tex**:
   ```bash
   /Users/kyle/Documents/ws/post-hoc-reasoning/render_latex.sh /Users/kyle/Documents/ws/post-hoc-reasoning/writing/nips-draft.tex
   ```

## Notes

- The script will automatically run pdflatex multiple times to resolve references
- It will run bibtex if bibliography commands are detected
- The PDF will automatically open in the default viewer on macOS
- Any compilation errors will be shown in the terminal output