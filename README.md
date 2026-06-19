<table>
<tr>
<td width="240" valign="top">
  <img src="assets/AIvory-logo.png" alt="AIvory logo" width="220">
</td>
<td valign="top">

# Ivory Agentic Repository (Extracted from Notebook)

A lightweight repository extracted from `Ivory_agentic.ipynb` for cleaner GitHub use, with separate ML and agentic workflows and reusable Python modules.

</td>
</tr>
</table>

## Structure

- `src/ivory_repo/ml.py`: ResNet34 + fastai segmentation utilities and inference helpers.
- `src/ivory_repo/llm.py`: Gemini and Anthropic client setup plus Gemini image classification helper.
- `examples/ml_resnet34_inference.ipynb`: Example notebook focused on ML segmentation inference.
- `examples/agentic_llm_workflows.ipynb`: Example notebook focused on LLM/agentic workflows.

## Quick Start

```bash
cd github_repo
python -m pip install -r requirements.txt
python -m pip install -e .
export GOOGLE_API_KEY="..."
export ANTHROPIC_API_KEY="..."
```

Use the notebooks in `examples/` for end-to-end demos.

## Notes

- The original `Ivory_agentic.ipynb` was not modified.
- API keys are loaded from environment variables, not hardcoded.
