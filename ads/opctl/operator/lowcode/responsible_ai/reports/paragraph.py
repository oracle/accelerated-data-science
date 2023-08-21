import os
from jinja2 import Environment, FileSystemLoader

template_dir = os.path.join(os.path.dirname(__file__), "templates")


def render_paragraph(text, metrics):
    sentences = [
        {"text": text[offset : offset + length], "scores": scores}
        for offset, length, scores in metrics
    ]
    env = Environment(loader=FileSystemLoader(template_dir))
    template = env.get_template("paragraph.html")
    html_string = template.render(sentences=sentences)
    return html_string


def test():
    s1 = (
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit, "
        "sed do eiusmod tempor incididunt ut labore et dolore magna aliqua."
    )
    s2 = (
        "Ut enim ad minim veniam, quis nostrud exercitation ullamco "
        "laboris nisi ut aliquip ex ea commodo consequat."
    )
    s3 = (
        "Duis aute irure dolor in reprehenderit in voluptate velit "
        "esse cillum dolore eu fugiat nulla pariatur."
    )
    s4 = (
        "Excepteur sint occaecat cupidatat non proident, "
        "sunt in culpa qui officia deserunt mollit anim id est laborum."
    )
    sentences = [s1, s2, s3, s4]
    text = " ".join(sentences)
    metrics = []
    offset = 0
    for s in sentences:
        metrics.append(
            (offset, len(s), {"score_A": len(s) / 100, "score_B": len(s) % 10 / 10})
        )
        offset += len(s) + 1
    html_string = render_paragraph(text=text, metrics=metrics)
    with open("rendered_paragraph.html", "w", encoding="utf-8") as f:
        f.write(html_string)


if __name__ == "__main__":
    test()
