"""Tests for VisualizerAgent code extraction edge cases."""

from __future__ import annotations

from pathlib import Path

from paperbanana.agents.visualizer import VisualizerAgent


class _DummyImageGen:
    async def generate(self, *args, **kwargs):
        return None


class _DummyVLM:
    async def generate(self, *args, **kwargs):
        return ""


def _make_agent(tmp_path):
    return VisualizerAgent(
        image_gen=_DummyImageGen(),
        vlm_provider=_DummyVLM(),
        prompt_dir=str(tmp_path),
        output_dir=str(tmp_path),
    )


def test_extract_code_handles_truncated_python_block(tmp_path):
    agent = _make_agent(tmp_path)
    response = "```python\nimport matplotlib.pyplot as plt\nplt.figure()\n"
    code = agent._extract_code(response)
    assert code == "import matplotlib.pyplot as plt\nplt.figure()"


def test_extract_code_handles_truncated_generic_block(tmp_path):
    agent = _make_agent(tmp_path)
    response = "```\nprint('hello')\n"
    code = agent._extract_code(response)
    assert code == "print('hello')"


def test_extract_code_handles_complete_python_block(tmp_path):
    agent = _make_agent(tmp_path)
    response = "```python\nprint('ok')\n```\nextra"
    code = agent._extract_code(response)
    assert code == "print('ok')"


def test_extract_code_handles_plain_code_response(tmp_path):
    agent = _make_agent(tmp_path)
    response = "import matplotlib.pyplot as plt\nplt.figure()"
    code = agent._extract_code(response)
    assert code == response


def test_execute_plot_code_produces_svg_vector_output(tmp_path):
    agent = _make_agent(tmp_path)
    output_path = str(tmp_path / "plot.png")
    code = (
        "import matplotlib.pyplot as plt\n"
        "fig, ax = plt.subplots()\n"
        "ax.plot([1, 2, 3], [4, 5, 6])\n"
        "fig.savefig(OUTPUT_PATH)\n"
    )
    success = agent._execute_plot_code(code, output_path, vector_format="svg")
    assert success
    assert Path(output_path).exists()
    assert (tmp_path / "plot.svg").exists()
    assert (tmp_path / "plot.svg").stat().st_size > 0


def test_execute_plot_code_produces_pdf_vector_output(tmp_path):
    agent = _make_agent(tmp_path)
    output_path = str(tmp_path / "plot.png")
    code = (
        "import matplotlib.pyplot as plt\n"
        "fig, ax = plt.subplots()\n"
        "ax.bar(['a', 'b'], [3, 7])\n"
        "fig.savefig(OUTPUT_PATH)\n"
    )
    success = agent._execute_plot_code(code, output_path, vector_format="pdf")
    assert success
    assert Path(output_path).exists()
    assert (tmp_path / "plot.pdf").exists()
    assert (tmp_path / "plot.pdf").stat().st_size > 0


def test_execute_plot_code_no_vector_when_not_requested(tmp_path):
    agent = _make_agent(tmp_path)
    output_path = str(tmp_path / "plot.png")
    code = (
        "import matplotlib.pyplot as plt\n"
        "fig, ax = plt.subplots()\n"
        "ax.plot([1, 2], [3, 4])\n"
        "fig.savefig(OUTPUT_PATH)\n"
    )
    success = agent._execute_plot_code(code, output_path)
    assert success
    assert Path(output_path).exists()
    assert not (tmp_path / "plot.svg").exists()
    assert not (tmp_path / "plot.pdf").exists()


def test_execute_plot_code_strips_vlm_vector_path_assignment(tmp_path):
    agent = _make_agent(tmp_path)
    output_path = str(tmp_path / "plot.png")
    code = (
        'VECTOR_PATH = "/tmp/hacked.svg"\n'
        "import matplotlib.pyplot as plt\n"
        "fig, ax = plt.subplots()\n"
        "ax.plot([1, 2], [3, 4])\n"
        "fig.savefig(OUTPUT_PATH)\n"
    )
    success = agent._execute_plot_code(code, output_path, vector_format="svg")
    assert success
    assert (tmp_path / "plot.svg").exists()
    assert not Path("/tmp/hacked.svg").exists()
