def md(text: str):
    return {"cell_type": "markdown", "metadata": {}, "source": _as_lines(text)}


def code(text: str):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _as_lines(text),
    }


def _as_lines(text: str) -> list[str]:
    text = text.strip("\n") + "\n"
    lines = text.splitlines(keepends=True)
    return lines
